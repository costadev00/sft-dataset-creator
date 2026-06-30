# Guia de execucao por maquina

Este guia lista comandos prontos para rodar o projeto em ambientes diferentes.
O fluxo atual usa apenas avaliacao deterministica; nao ha `--judge-model`,
`--audit-fraction` nem troca de modelo para avaliacao.

## Pre-checks

Dentro do repositorio:

```bash
source .venv/bin/activate
.venv/bin/sft-dataset plugins
.venv/bin/sft-dataset run --help
```

Em maquinas com NVIDIA:

```bash
nvidia-smi
```

Para a nossa maquina principal, espere 4 GPUs RTX 4000 Ada ou equivalentes. O
default do projeto usa `tensor_parallel_size=4`.

## Perfil A: CPU, CI ou desenvolvimento rapido

Use o backend `fake` para testar o pipeline inteiro sem carregar LLM:

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
  --run-dir runs/dev-fake-20
```

Esse modo deve terminar rapido e gerar:

```text
runs/dev-fake-20/report.json
runs/dev-fake-20/exports/messages/train.jsonl
runs/dev-fake-20/exports/alpaca/train.jsonl
runs/dev-fake-20/exports/prompt_completion/train.jsonl
```

## Perfil B: API externa sem GPU

Instale o extra `openai` e use o backend `openai_compatible`:

```bash
python -m pip install -e '.[dev,hf,openai]'
export OPENAI_API_KEY=<sua-chave>
```

Run:

```bash
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
  --generator-param timeout=180 \
  --run-dir runs/api-100
```

Se o endpoint local nao suportar `json_schema`, o backend tenta fallback para
chat completions comum.

## Perfil C: 1 GPU ou GPU com pouca VRAM

Nao use o Gemma 31B como primeira opcao nesse perfil. Escolha um modelo menor e
reduza o tamanho dos lotes:

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
  --run-dir runs/single-gpu-50
```

Se ainda faltar memoria:

```bash
--generator-param max_num_batched_tokens=2048
--generator-param max_num_seqs=2
--chunk-size 4000
```

Use `quantization=compressed-tensors` apenas se o checkpoint escolhido realmente
foi publicado nesse formato.

## Perfil D: nossa maquina 4x RTX 4000 Ada

Instalacao:

```bash
python -m pip install -e '.[dev,hf,local]'
```

Smoke real com 20 documentos:

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

Run intermediaria:

```bash
.venv/bin/sft-dataset run \
  --dataset costadev00/wiki-brazil \
  --examples 200 \
  --documents 300 \
  --language pt-BR \
  --id-field __row_index__ \
  --profile wikipedia_ptbr \
  --run-dir runs/wiki-brazil-200
```

Run maior:

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

Parametros aplicados automaticamente ao Gemma 31B QAT:

```text
tensor_parallel_size=4
quantization=compressed-tensors
kv_cache_dtype=fp8
max_num_batched_tokens=16384
download_dir=<cache>/models
```

Parametros opcionais para experimentar throughput:

```bash
--generator-param max_num_seqs=16
--generator-param enable_chunked_prefill=true
--generator-param enable_prefix_caching=true
--generator-param gpu_memory_utilization=0.92
```

## Run de 20% da `wikipedia-pt-br-extract`

Para rodar 20% dos documentos elegiveis do dataset
`costadev00/wikipedia-pt-br-extract`, fixe a revisao e conte os elegiveis antes
de chamar `run`. Esse dataset pode ser gated no Hugging Face; a maquina precisa
estar autenticada e com o acesso aceito.

Preparacao:

```bash
cd /home/matheuscm/sft-dataset-creator
source .venv/bin/activate
python -m pip install -e '.[hf,local,dashboard]'
hf auth login || huggingface-cli login

export DATASET_ID="costadev00/wikipedia-pt-br-extract"
export DATASET_REVISION="cdbd07dc4a3de6e64632c718710b3ae0ebaeb0ff"
```

Preflight para calcular `SELECTED_DOCS` e `RECOMMENDED_EXAMPLES`:

```bash
mkdir -p runs
.venv/bin/sft-dataset preflight \
  --dataset "$DATASET_ID" \
  --dataset-revision "$DATASET_REVISION" \
  --source huggingface \
  --split train \
  --streaming \
  --profile wikipedia_ptbr \
  --id-field page_id \
  --text-field text \
  --title-field title \
  --sections-field section_texts \
  --license-field license \
  --selection-fraction 0.2 \
  --per-document-min 14 \
  --per-document-max 14 \
  --progress-every 100000 \
  --shell | tee runs/wiki-ptbr-extract-20pct.preflight.env

source runs/wiki-ptbr-extract-20pct.preflight.env
export EXAMPLES="$RECOMMENDED_EXAMPLES"
```

Argumentos comuns da distribuicao:

```bash
TASK_ARGS=(
  --task classification=1
  --task closed_qa=1
  --task comparison=1
  --task concept_explanation=1
  --task definition=1
  --task didactic_explanation=1
  --task fact_checking=1
  --task information_extraction=1
  --task rewrite=1
  --task short_answer=1
  --task structured_extraction=1
  --task summarization=1
  --task taxonomy=1
  --task timeline=1
)

DIFFICULTY_ARGS=(
  --difficulty easy=0.25
  --difficulty medium=0.5
  --difficulty hard=0.25
)
```

Calibracao com 5.000 documentos, mantendo 14 slots por documento:

```bash
export CALIB_DOCS=5000
export CALIB_EXAMPLES=$((CALIB_DOCS * 14))
export CALIB_RUN_DIR="runs/wiki-ptbr-extract-calib-5kdocs-14tasks"

.venv/bin/sft-dataset run \
  --dataset "$DATASET_ID" \
  --dataset-revision "$DATASET_REVISION" \
  --source huggingface \
  --split train \
  --streaming \
  --language pt-BR \
  --profile wikipedia_ptbr \
  --id-field page_id \
  --text-field text \
  --title-field title \
  --sections-field section_texts \
  --license-field license \
  --documents "$CALIB_DOCS" \
  --examples "$CALIB_EXAMPLES" \
  --reserve-fraction 0 \
  --per-document-min 14 \
  --per-document-max 14 \
  --max-attempts 5 \
  --attempt-multiplier 3 \
  "${TASK_ARGS[@]}" \
  "${DIFFICULTY_ARGS[@]}" \
  --generator-plugin vllm_local \
  --model google/gemma-4-31B-it-qat-w4a16-ct \
  --formats messages,prompt_completion,alpaca \
  --containers jsonl \
  --run-dir "$CALIB_RUN_DIR" \
  --allow-partial
```

Run principal:

```bash
export RUN_DIR="runs/wiki-ptbr-extract-20pct-14tasks"

.venv/bin/sft-dataset run \
  --dataset "$DATASET_ID" \
  --dataset-revision "$DATASET_REVISION" \
  --source huggingface \
  --split train \
  --streaming \
  --language pt-BR \
  --profile wikipedia_ptbr \
  --id-field page_id \
  --text-field text \
  --title-field title \
  --sections-field section_texts \
  --license-field license \
  --selection-fraction 0.2 \
  --examples "$EXAMPLES" \
  --reserve-fraction 0 \
  --per-document-min 14 \
  --per-document-max 14 \
  --max-attempts 5 \
  --attempt-multiplier 3 \
  "${TASK_ARGS[@]}" \
  "${DIFFICULTY_ARGS[@]}" \
  --generator-plugin vllm_local \
  --model google/gemma-4-31B-it-qat-w4a16-ct \
  --formats messages,prompt_completion,alpaca \
  --containers jsonl \
  --run-dir "$RUN_DIR" \
  --allow-partial
```

Acompanhamento:

```bash
.venv/bin/sft-dataset status "$RUN_DIR" --watch --interval 10
.venv/bin/sft-dataset dashboard "$RUN_DIR" --host 127.0.0.1 --port 8765
```

De outra maquina local, tunel SSH:

```bash
ssh -L 8765:127.0.0.1:8765 usuario@IP_DA_S10
```

Abra `http://127.0.0.1:8765` no navegador local.

Finalizacao:

```bash
.venv/bin/sft-dataset checkpoint "$RUN_DIR"
.venv/bin/sft-dataset export "$RUN_DIR"
cat "$RUN_DIR/report.json"
```

## Perfil E: mais ou menos GPUs que a maquina principal

Com 2 GPUs:

```bash
--generator-param tensor_parallel_size=2
```

Com 8 GPUs:

```bash
--generator-param tensor_parallel_size=8
--generator-param max_num_batched_tokens=32768
```

Esses ajustes nao garantem que qualquer modelo cabera. Eles apenas alinham o
tensor parallel ao hardware. Sempre comece com `--smoke-models`.

## Cache e discos

Use um cache persistente em disco rapido quando possivel:

```bash
--cache-dir /mnt/ssd/sft-cache
```

O comando configura internamente:

```text
HF_HOME=<cache>/huggingface
HF_HUB_CACHE=<cache>/huggingface/hub
HF_XET_CACHE=<cache>/huggingface/xet
TRANSFORMERS_CACHE=<cache>/huggingface/transformers
```

## Planejar e executar em etapas

O uso direto com `run --dataset ...` e o caminho mais simples. Para pipelines
mais controlados, use um arquivo de config:

```bash
.venv/bin/sft-dataset validate --config sft-project.json
.venv/bin/sft-dataset plan --config sft-project.json --run-dir runs/minha-run
.venv/bin/sft-dataset run --plan runs/minha-run/plan.json
```

O plano so executa se o hash de `config.resolved.json` continuar compativel.

## Retomar uma run

```bash
.venv/bin/sft-dataset status runs/wiki-brazil-1000
.venv/bin/sft-dataset run --resume runs/wiki-brazil-1000
```

Retomada avalia tentativas geradas e nao avaliadas antes de chamar o modelo
novamente.

## Conferir resultado

```bash
.venv/bin/sft-dataset inspect runs/wiki-brazil-1000 --limit 10
cat runs/wiki-brazil-1000/report.json
ls runs/wiki-brazil-1000/exports/*/
```

Exports padrao:

```text
messages/train.jsonl
prompt_completion/train.jsonl
alpaca/train.jsonl
```

## Quando usar `--smoke-models`

Use:

- primeira execucao em uma maquina nova;
- depois de trocar modelo, vLLM, driver ou CUDA;
- antes de uma run grande;
- depois de mudar parametros de quantizacao, tensor parallel ou KV cache.

Evite:

- toda retomada curta;
- testes com backend `fake`;
- loops de desenvolvimento que nao precisam carregar o modelo.

## Ordem sugerida para producao

1. Rode backend `fake` com 20 exemplos.
2. Rode smoke real com `--smoke-models` e 20 exemplos.
3. Rode uma amostra intermediaria com 100 a 200 exemplos.
4. Inspecione exemplos aceitos.
5. Ajuste tarefas, dificuldades ou prompt se necessario.
6. Rode a meta final.
7. Crie uma auditoria manual antes de publicar.

## Publicacao

```bash
export HF_TOKEN=<token-com-permissao>
.venv/bin/sft-dataset publish runs/wiki-brazil-1000 --repo-id owner/dataset
```

Por padrao a publicacao e privada. Use `--public` somente quando o dataset ja
tiver sido revisado.
