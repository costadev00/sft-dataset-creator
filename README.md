# sft-dataset-creator

`sft-dataset-creator` e uma biblioteca e CLI para planejar, gerar, avaliar,
exportar e publicar datasets sinteticos de SFT com rastreabilidade. O projeto
foi ajustado para o fluxo atual:

- geracao com um unico modelo por run;
- modelo local padrao: `google/gemma-4-31B-it-qat-w4a16-ct`;
- vLLM local com `compressed-tensors` e FP8 no KV cache;
- avaliacao automatica e deterministica, sem judge model;
- exports `train`-only por padrao;
- runs retomaveis por `run.db`, `plan.json` e `config.resolved.json`.

## Fluxo

```text
source scan -> deterministic plan -> continuous batched generation
            -> chunked document context -> deterministic quality gates
            -> grouped exports -> optional audit -> optional Hub publication
```

Cada run salva a configuracao resolvida, snapshot do corpus, plano imutavel,
ledger SQLite, relatorio e exports finais. Se a execucao cair, `run --resume`
continua dos slots ainda nao aceitos.

## Instalacao

Ambiente base para desenvolvimento, testes, fontes locais e Hugging Face:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev,hf]'
```

Para inferencia local com GPU/vLLM:

```bash
python -m pip install -e '.[dev,hf,local]'
```

Para backend HTTP compativel com OpenAI:

```bash
python -m pip install -e '.[dev,hf,openai]'
```

## Escolha do modo de execucao

Use este quadro como decisao rapida:

| Maquina | Backend recomendado | Uso |
| --- | --- | --- |
| CPU ou notebook sem GPU | `fake` | testar pipeline, planner, exports e auditoria sem custo de modelo |
| Sem GPU, com API externa | `openai_compatible` | gerar dados reais chamando um endpoint HTTP |
| 1 GPU ou pouca VRAM | `vllm_local` com modelo menor | smoke real local, batches pequenos, menor throughput |
| 4x RTX 4000 Ada ou equivalente | `vllm_local` padrao | caminho principal com Gemma 31B QAT e FP8 KV cache |

O default da CLI e otimizado para a maquina com 4 GPUs:

```text
model=google/gemma-4-31B-it-qat-w4a16-ct
tensor_parallel_size=4
quantization=compressed-tensors
kv_cache_dtype=fp8
max_num_batched_tokens=16384
download_dir=<cache>/models
```

Esses parametros sao aplicados automaticamente quando o backend e
`vllm_local` e o modelo e o Gemma 31B QAT acima. Qualquer valor passado com
`--generator-param KEY=VALUE` tem prioridade.

## Smoke recomendado na maquina 4x RTX 4000 Ada

Antes de uma run grande, rode um smoke pequeno com carregamento real do modelo:

```bash
.venv/bin/sft-dataset run \
  --dataset costadev00/wiki-brazil \
  --examples 20 \
  --documents 20 \
  --language pt-BR \
  --id-field __row_index__ \
  --profile wikipedia_ptbr \
  --smoke-models \
  --run-dir runs/wiki-brazil-smoke-20
```

`--smoke-models` carrega o checkpoint e envia uma requisicao JSON curta antes
do planejamento. Use em smoke e pre-producao; em retomadas pequenas, pode omitir
para economizar tempo.

## Run maior na maquina 4x RTX 4000 Ada

Exemplo com 1.000 exemplos finais, mantendo os defaults do Gemma 31B QAT:

```bash
.venv/bin/sft-dataset run \
  --dataset costadev00/wiki-brazil \
  --examples 1000 \
  --documents 1500 \
  --language pt-BR \
  --id-field __row_index__ \
  --profile wikipedia_ptbr \
  --run-dir runs/wiki-brazil-1000
```

Notas:

- `--examples` e o numero final de exemplos aceitos no dataset.
- `--documents` limita quantos documentos entram no snapshot da run.
- `--profile wikipedia_ptbr` filtra namespaces, redirects, desambiguacao e
  textos muito curtos.
- o split padrao e `train=1.0`, portanto os exports geram apenas `train.jsonl`.
- a avaliacao e deterministica-only; nao existe segundo modelo de avaliacao.

## Maquina com 1 GPU ou pouca VRAM

O Gemma 31B QAT foi pensado para a nossa maquina multi-GPU. Em uma unica GPU,
prefira um modelo menor e reduza o lote:

```bash
.venv/bin/sft-dataset run \
  --dataset costadev00/wiki-brazil \
  --examples 50 \
  --documents 80 \
  --language pt-BR \
  --id-field __row_index__ \
  --profile wikipedia_ptbr \
  --model <modelo-menor-instruct> \
  --generator-param tensor_parallel_size=1 \
  --generator-param max_num_seqs=4 \
  --generator-param max_num_batched_tokens=4096 \
  --generator-param gpu_memory_utilization=0.90 \
  --generator-param enable_chunked_prefill=true \
  --generator-param enable_prefix_caching=true \
  --run-dir runs/wiki-brazil-small-gpu
```

Se o modelo escolhido nao usa `compressed-tensors`, nao passe
`quantization=compressed-tensors`. Se houver OOM, reduza primeiro
`max_num_batched_tokens`, depois `max_num_seqs`, e por fim `--chunk-size`.

## CPU ou teste sem modelo

Para validar rapidamente planner, SQLite, avaliacao deterministica e exports:

```bash
.venv/bin/sft-dataset run \
  --dataset costadev00/wiki-brazil \
  --examples 20 \
  --documents 20 \
  --language pt-BR \
  --id-field __row_index__ \
  --profile wikipedia_ptbr \
  --generator-plugin fake \
  --model fake-generator \
  --run-dir runs/wiki-brazil-fake-smoke
```

Esse modo nao mede qualidade de linguagem; ele serve para testar o encadeamento
do projeto.

## API externa sem GPU

Use `openai_compatible` para apontar para um endpoint HTTP que implemente chat
completions. O nome do modelo e os parametros dependem do servidor usado:

```bash
export OPENAI_API_KEY=<sua-chave>

.venv/bin/sft-dataset run \
  --dataset costadev00/wiki-brazil \
  --examples 100 \
  --documents 150 \
  --language pt-BR \
  --id-field __row_index__ \
  --profile wikipedia_ptbr \
  --generator-plugin openai_compatible \
  --model <modelo-do-endpoint> \
  --generator-param base_url=https://api.openai.com/v1 \
  --generator-param api_key_env=OPENAI_API_KEY \
  --run-dir runs/wiki-brazil-api-100
```

Para servidores locais compativeis, troque `base_url` para o endpoint local.

## Fontes locais

JSONL local:

```bash
.venv/bin/sft-dataset run \
  --dataset data/corpus.jsonl \
  --source local \
  --source-format jsonl \
  --examples 100 \
  --generator-plugin fake \
  --model fake-generator \
  --run-dir runs/local-fake
```

Campos comuns podem ser remapeados:

```bash
--id-field id --text-field text --title-field title --sections-field sections
```

Formatos suportados para fonte local: JSONL/NDJSON, JSON e Parquet.

## Retomar, inspecionar e exportar

```bash
.venv/bin/sft-dataset run --resume runs/wiki-brazil-1000
.venv/bin/sft-dataset status runs/wiki-brazil-1000
.venv/bin/sft-dataset inspect runs/wiki-brazil-1000 --limit 10
.venv/bin/sft-dataset export runs/wiki-brazil-1000
```

Se a run ficar `partial`, os artefatos ainda ficam no diretorio da run. Para
publicar, complete a meta ou revise explicitamente o uso do dataset parcial.

## Splits e formatos de saida

Por padrao:

```text
formats=messages,prompt_completion,alpaca
containers=jsonl
train=1.0
validation=0.0
test=0.0
```

Para gerar validacao e teste:

```bash
--train-split 0.9 --validation-split 0.05 --test-split 0.05
```

Os arquivos ficam em:

```text
runs/<run-id>/exports/messages/train.jsonl
runs/<run-id>/exports/prompt_completion/train.jsonl
runs/<run-id>/exports/alpaca/train.jsonl
```

## Auditoria manual

Depois de gerar exemplos, crie uma amostra cega:

```bash
.venv/bin/sft-dataset audit-sample runs/wiki-brazil-1000 --size 300
```

Preencha `runs/wiki-brazil-1000/audit/review.jsonl` com `human_verdict` igual a
`accept`, `reject` ou `review`, depois rode:

```bash
.venv/bin/sft-dataset audit-score runs/wiki-brazil-1000
```

Campos legados como `llm_route_rate` permanecem para compatibilidade e devem
ficar em zero no fluxo deterministico-only.

## Tuning de batching

O tuning atual otimiza apenas a geracao:

```bash
.venv/bin/sft-dataset tune \
  --config runs/wiki-brazil-1000/config.resolved.json \
  --output runs/wiki-brazil-1000/config.tuned.json \
  --stage generation \
  --samples 32
```

Use com cuidado: tuning local carrega modelo e executa requisicoes reais.

## Reprodutibilidade

- Passe `--dataset-revision <commit>` em runs de producao no Hugging Face.
- Use `--cache-dir` em disco rapido e persistente.
- O projeto grava `config.resolved.json`, `plan.json` e `corpus-selected.jsonl`.
- O hash da configuracao protege contra executar um plano com outra config.
- Segredos ficam em variaveis de ambiente; nao sao gravados na config resolvida.

## Troubleshooting rapido

- `fewer GPUs detected than generation.tensor_parallel_size`: reduza
  `--generator-param tensor_parallel_size=<n>` ou use uma maquina com GPUs
  suficientes.
- OOM no vLLM: reduza `max_num_batched_tokens`, `max_num_seqs` e `--chunk-size`,
  ou use modelo menor.
- Modelo gated no Hugging Face: autentique com `hf auth login` ou defina
  `HF_TOKEN`.
- `--smoke-models` lento: esperado, ele carrega o checkpoint antes da run.
- Muitos rejeitados: inspecione `report.json`, `run.db` e exemplos com
  `sft-dataset inspect`.

## Desenvolvimento

```bash
.venv/bin/pytest
```

Testes CPU usam o backend `fake`. Smoke GPU e benchmarks devem ser executados na
maquina-alvo antes de uma geracao grande.

Mais detalhes de execucao estao em [docs/running.md](docs/running.md) e o desenho
interno esta em [docs/architecture.md](docs/architecture.md).
