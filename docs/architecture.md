# Arquitetura

Este documento descreve como o `sft-dataset-creator` organiza uma run de SFT.
Para comandos por tipo de maquina, veja [running.md](running.md).

## Contratos publicos

A superficie Python principal e composta por:

- `ProjectConfig`
- `BatchingConfig`
- `BatchGenerationResult`
- `load_config`
- `build_plan`
- `execute_plan`
- `tune_project`
- `export_run`
- `publish_run`

Os registros sao modelos Pydantic estritos. Chaves desconhecidas falham cedo,
antes de qualquer chamada a modelo.

A CLI `sft-dataset run` monta um `ProjectConfig` diretamente a partir das
opcoes de linha de comando. O arquivo `config.resolved.json` e um artefato de
execucao, nao um arquivo que o usuario precise escrever manualmente.

## Extensoes

Os pontos de extensao sao entry points Python:

- `sft_dataset_creator.sources`
- `sft_dataset_creator.tasks`
- `sft_dataset_creator.backends`
- `sft_dataset_creator.evaluators`
- `sft_dataset_creator.exporters`

Um source emite `Document`. Uma task transforma um slot planejado e um documento
em `GenerationRequest`. Um backend retorna JSON estruturado. Um evaluator opera
sobre `SFTCandidate`. Um exporter converte exemplos aceitos para uma visao final
do dataset.

## Backends

Backends inclusos:

| Backend | Uso | Observacao |
| --- | --- | --- |
| `fake` | testes CPU/CI | nao chama LLM real |
| `openai_compatible` | API externa | usa chat completions HTTP |
| `vllm_local` | GPU local | usa vLLM em subprocesso isolado |

O backend local com vLLM encaminha parametros via `--generator-param`, incluindo:

- `tensor_parallel_size`
- `gpu_memory_utilization`
- `quantization`
- `kv_cache_dtype`
- `max_num_seqs`
- `max_num_batched_tokens`
- `enable_chunked_prefill`
- `enable_prefix_caching`
- `download_dir`
- `kv_cache_memory_bytes`
- `cpu_offload_gb`

Para o modelo padrao `google/gemma-4-31B-it-qat-w4a16-ct`, a CLI aplica:

```text
tensor_parallel_size=4
quantization=compressed-tensors
kv_cache_dtype=fp8
max_num_batched_tokens=16384
download_dir=<cache>/models
```

Esse e o perfil principal para a maquina 4x RTX 4000 Ada.

## Planejamento

O planejamento nao chama modelo. Ele:

1. carrega a fonte Hugging Face ou local;
2. normaliza registros em `Document`;
3. aplica filtros de perfil, por exemplo `wikipedia_ptbr`;
4. valida IDs unicos;
5. seleciona documentos por contagem ou fracao;
6. cria chunks deterministico de texto;
7. distribui tarefas e dificuldades;
8. grava `plan.json`, `corpus-index.jsonl` e `corpus-selected.jsonl`.

`plan.json` e amarrado a `config.resolved.json` por SHA-256. A execucao rejeita
um plano se a config resolvida nao bater com o hash do plano.

## Chunking

Cada request de geracao recebe um chunk planejado, nao o documento inteiro. Isso
controla custo de contexto e mantem offsets de evidencia verificaveis.

Parametros relevantes:

- `--chunk-size`, default `8000`
- `--chunk-overlap`, default `400`

Se uma run precisa reduzir memoria ou latencia, diminuir `--chunk-size` costuma
ser mais seguro do que alterar prompt ou schema.

## Execucao

`run.db` e a fonte transacional da verdade. Ele guarda slots, tentativas,
respostas, avaliacoes e exemplos aceitos.

O motor executa rodadas:

1. identifica slots pendentes;
2. monta requests da rodada;
3. envia requests ao backend de geracao;
4. grava candidatos gerados;
5. aplica avaliacao deterministica;
6. aceita, rejeita ou marca para nova tentativa;
7. repete ate atingir a meta ou esgotar limites.

Em vLLM local, o gerador fica carregado entre rodadas. Nao ha segundo modelo de
avaliacao, portanto nao ha troca de checkpoint na GPU.

## Avaliacao deterministica

O fluxo atual removeu judge LLM. A avaliacao e sempre automatica e
deterministica.

Gates principais:

- instrucao e resposta nao vazias;
- tamanho minimo de instrucao;
- evidencia presente;
- offsets de evidencia validos;
- quote consistente quando fornecida;
- rejeicao de referencias ao texto oculto, como "de acordo com o texto";
- rejeicao de duplicatas normalizadas;
- validacao final no exporter para impedir vazamento de fonte oculta.

Campos legados ainda existem para compatibilidade:

- `EvaluationConfig.llm`
- `RoutingConfig`
- `AcceptanceConfig`
- `llm_judged_examples`
- `llm_judge_coverage`
- `semantic_judge_configured`

Mesmo se uma config antiga contiver `evaluation.llm`, o engine ignora esse campo.
Os relatorios novos devem indicar `llm_judged_examples=0` e
`semantic_judge_configured=false`.

## Retomada

Uma run interrompida pode ser retomada com:

```bash
sft-dataset run --resume runs/<run-id>
```

Na retomada, tentativas ja geradas e ainda nao avaliadas sao avaliadas antes de
novas chamadas de geracao. Slots aceitos nao sao regenerados.

## Limites e parciais

Os limites principais sao:

- `--max-attempts`, tentativas por slot;
- `--attempt-multiplier`, limite global de tentativas;
- `--reserve-fraction`, documentos extras para substituicao;
- `--same-document-attempts`, quantas tentativas usam o documento original antes
  de recorrer a reserva.

Se a meta nao for atingida, a run termina como `partial`. Os artefatos continuam
uteis para diagnostico, mas publicacao direta e bloqueada pela CLI.

## Exports

Views inclusas:

- `messages`
- `prompt_completion`
- `alpaca`

Container padrao:

- `jsonl`

Splits padrao:

```text
train=1.0
validation=0.0
test=0.0
```

O exporter remove arquivos stale de splits desabilitados e grava
`generation_info.json` com gerador, evaluator deterministico e hash da config.

## Auditoria manual

`audit-sample` cria uma amostra cega de exemplos avaliados pelo sistema.
`audit-score` compara os rotulos humanos com as decisoes deterministicas.

O campo `llm_route_rate` pode aparecer no relatorio de auditoria por
compatibilidade, mas no fluxo atual deve permanecer em zero.

## Doctor e smoke

`collect_doctor_report` verifica:

- versao Python e plataforma;
- GPUs via `nvidia-smi`;
- plugins disponiveis;
- pacotes instalados;
- permissao de escrita em caminhos;
- compatibilidade basica entre `tensor_parallel_size` e GPUs detectadas.

Com `--smoke-models`, a CLI carrega apenas o gerador configurado e solicita uma
resposta JSON minima. Isso valida o caminho de modelo sem iniciar uma run grande.

## Tuning

`tune_project` otimiza somente a etapa de geracao. Ele compara perfis sequencial
e async, ajustando parametros como:

- `max_num_seqs`
- `max_num_batched_tokens`
- `gpu_memory_utilization`
- `enable_chunked_prefill`
- `enable_prefix_caching`

Como tuning carrega o modelo e executa requests reais, deve ser usado na maquina
alvo.

## Estrutura de uma run

```text
runs/<run-id>/
  config.resolved.json
  manifest.json
  plan.json
  corpus-index.jsonl
  corpus-selected.jsonl
  run.db
  report.json
  exports/
    messages/train.jsonl
    prompt_completion/train.jsonl
    alpaca/train.jsonl
```

`manifest.json` registra ambiente, plugins, pacotes e GPUs. `report.json`
resume status, contagens, metricas de geracao, tokens e deficits.

## Sequencia recomendada

1. Validar o pipeline com backend `fake`.
2. Rodar smoke real com `--smoke-models`.
3. Rodar uma amostra intermediaria.
4. Inspecionar exemplos aceitos.
5. Ajustar prompts, composicao ou parametros de batching.
6. Rodar a meta final.
7. Criar auditoria manual.
8. Publicar somente depois de revisao.
