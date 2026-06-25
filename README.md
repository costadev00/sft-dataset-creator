# sft-dataset-creator

`sft-dataset-creator` is a Python library and CLI for planning, generating, evaluating, exporting, and publishing reproducible synthetic supervised fine-tuning datasets.

The framework is source-agnostic and model-agnostic. A run is declared directly through CLI options: corpus selection, final example count, task and difficulty composition, replacement limits, model backends, and output views. The planner turns those options into a validated, immutable execution plan before any generation call is made.

## Core workflow

```text
source scan -> deterministic plan -> continuous batched generation
            -> iterative document chunks -> quality gates
            -> grouped splits -> exports -> optional Hub publication
```

Every run stores its resolved configuration, selected corpus snapshot, immutable plan, transactional SQLite ledger, report, and final exports. Runs can be resumed without repeating accepted slots.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,hf]'
```

For local GPU inference:

```bash
python -m pip install -e '.[dev,hf,local]'
```

The application runs directly as Python processes and does not require Docker.
Local vLLM models run in isolated subprocesses on the configured GPUs.

## Editing prompts

All prompts sent to generation and model smoke-test requests are kept in
[sft_dataset_creator/prompts.py](sft_dataset_creator/prompts.py). Edit that file
to change model behavior without navigating execution or backend code.

## CLI

Run directly from a Hugging Face dataset id. No input configuration file is required:

```bash
sft-dataset run \
  --dataset costadev00/wiki-brazil \
  --examples 1000 \
  --language pt-BR \
  --id-field __row_index__ \
  --profile wikipedia_ptbr \
  --generator-param tensor_parallel_size=4 \
  --smoke-models \
  --run-dir runs/wiki-brazil
```

`huggingface`, source split `train`, streaming mode, the Gemma 31B QAT generator, deterministic-only evaluation, JSONL exports, and train-only output exports are defaults. Use `sft-dataset run --help` for all source, selection, model, retry, and output options.

For reproducible production runs, pass a real Hugging Face commit SHA with `--dataset-revision`. Omit the option for quick smoke tests; placeholder values are rejected.

Local JSON, JSONL, and Parquet corpora use the same command:

```bash
sft-dataset run \
  --dataset corpus.jsonl \
  --source local \
  --examples 100 \
  --generator-plugin fake \
  --model fake-generator
```

Repeat weighted options and backend parameters as needed:

```bash
--task closed_qa=0.6 --task summarization=0.4
--difficulty easy=0.25 --difficulty medium=0.5 --difficulty hard=0.25
--generator-param max_num_seqs=16 --generator-param enable_prefix_caching=true
```

Run management remains non-interactive and automation-safe:

```bash
sft-dataset run --resume runs/<run-id>
sft-dataset status runs/<run-id>
sft-dataset inspect runs/<run-id> --limit 10
sft-dataset audit-sample runs/<run-id> --size 300
sft-dataset audit-score runs/<run-id>
sft-dataset export runs/<run-id>
sft-dataset publish runs/<run-id> --repo-id owner/dataset
```

Every run writes `config.resolved.json`. This is an output artifact used for hashing, auditing, export, and resume; users do not need to create it.

`--smoke-models` is intentionally opt-in because it loads configured checkpoints. Environment and dependency checks run before the dataset scan; the smoke request additionally validates model loading and structured JSON output.

## Built-in integrations

- Sources: Hugging Face Datasets and local JSON, JSONL, or Parquet.
- Backends: isolated local vLLM, OpenAI-compatible HTTP, and a deterministic fake backend.
- Tasks: question answering, summarization, extraction, classification, explanations, fact checking, timelines, rewrites, and related SFT recipes.
- Evaluation: deterministic schema, evidence, and self-contained-content gates.
- Outputs: chat messages, prompt/completion, and Alpaca views in JSONL or Parquet.
- Publication: generic private-by-default Hugging Face Hub upload.

Third-party packages can register sources, task recipes, backends, evaluators, and exporters through Python entry points. See [the architecture guide](docs/architecture.md).

## Local GPU Example

The default generator is `google/gemma-4-31B-it-qat-w4a16-ct`. For local vLLM runs, the CLI automatically applies the 31B QAT defaults:

```bash
tensor_parallel_size=4
quantization=compressed-tensors
kv_cache_dtype=fp8
max_num_batched_tokens=16384
download_dir=<cache>/models
```

Use `--smoke-models` before large runs to validate that the configured generator loads and returns structured JSON on the target machine. Evaluation is automatic and deterministic; there is no judge model or GPU model swap.

Inside the generator process, vLLM `AsyncLLM` receives a bounded stream of concurrent requests and performs continuous batching. You can still add generation backend parameters with repeated `--generator-param KEY=VALUE` options when you need to override the defaults.

Documents are split into bounded, overlapping character chunks during planning.
When a document receives multiple examples, its slots iterate over chunk IDs in
order before cycling back. Configure this with `target.chunk_size_characters`
and `target.chunk_overlap_characters` internally, exposed as `--chunk-size` and
`--chunk-overlap` on the CLI.

## Reproducibility

CLI values are converted into strict Pydantic models. Unknown task names, invalid distributions, incompatible split ratios, and invalid token or chunk budgets fail before generation. The resolved configuration is stored in the run directory and tied to `plan.json` by SHA-256.

Advanced integrations can still create and validate `ProjectConfig` JSON programmatically. Generate its schema with:

```bash
sft-dataset schema --output sft-project.schema.json
```

Secrets are referenced by environment variable names and are never stored in resolved run configurations. The default Hugging Face publication token variable is `HF_TOKEN`.

## Development

```bash
pytest
```

CPU tests use the fake backend. GPU model loading is covered by opt-in smoke tests and should be run on the target cluster before production.
