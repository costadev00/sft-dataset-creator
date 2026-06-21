# sft-dataset-creator

`sft-dataset-creator` is a Python library and CLI for planning, generating, evaluating, exporting, and publishing reproducible synthetic supervised fine-tuning datasets.

The framework is source-agnostic and model-agnostic. A project describes the desired corpus selection, final example count, task and difficulty composition, grounding policy, replacement limits, model backends, and output views. The planner turns that specification into an immutable execution plan before any generation call is made.

## Core workflow

```text
source scan -> deterministic plan -> continuous batched generation
            -> iterative document chunks -> quality gates
            -> batched selective LLM evaluation
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

All prompts sent to generation, judge, and model smoke-test requests are kept in
[sft_dataset_creator/prompts.py](sft_dataset_creator/prompts.py). Edit that file
to change model behavior without navigating execution or backend code.

## CLI

Create a configuration interactively:

```bash
sft-dataset wizard --output project.json
```

The remaining commands are non-interactive and automation-safe:

```bash
sft-dataset validate --config project.json
sft-dataset doctor --config project.json
sft-dataset tune --config project.json --output project.tuned.json
sft-dataset plan --config project.json
sft-dataset run --config project.json
sft-dataset run --resume runs/<run-id>
sft-dataset status runs/<run-id>
sft-dataset inspect runs/<run-id> --limit 10
sft-dataset audit-sample runs/<run-id> --size 300
sft-dataset audit-score runs/<run-id>
sft-dataset export runs/<run-id>
sft-dataset publish runs/<run-id> --repo-id owner/dataset
```

`doctor --smoke-models` loads configured models sequentially and performs a structured-output request. It can download large checkpoints and is intentionally opt-in.

`tune` benchmarks a serial baseline and progressively larger async profiles. It writes the selected limits into a new reproducible configuration and records throughput, latency, VRAM, GPU utilization, package versions, and failures in a sibling tuning report.

## Built-in integrations

- Sources: Hugging Face Datasets and local JSON, JSONL, or Parquet.
- Backends: isolated local vLLM, OpenAI-compatible HTTP, and a deterministic fake backend.
- Tasks: question answering, summarization, extraction, classification, explanations, fact checking, timelines, rewrites, and related SFT recipes.
- Evaluation: deterministic schema, evidence, and self-contained-content gates plus selective LLM judging.
- Outputs: chat messages, prompt/completion, and Alpaca views in JSONL or Parquet.
- Publication: generic private-by-default Hugging Face Hub upload.

Third-party packages can register sources, task recipes, backends, evaluators, and exporters through Python entry points. See [the architecture guide](docs/architecture.md).

## Cluster preset

[examples/gemma-wikipedia.json](examples/gemma-wikipedia.json) targets four NVIDIA RTX 4000 Ada GPUs:

- Generator: `google/gemma-4-26B-A4B-it`, BF16, tensor parallel 4.
- Selective evaluator: `google/gemma-4-31B-it-qat-w4a16-ct`, tensor parallel 4.
- Context window: 64k with a 52k input budget.
- Evaluation routing: hard tasks, truncated context, high risk, weak grounding, and a deterministic 10% audit sample.

Generation and evaluation run in separate spawned processes so the first model releases GPU memory before the second is loaded. Inside each process, vLLM `AsyncLLM` receives a bounded stream of concurrent requests and performs continuous batching. Prefix caching and chunked prefill are enabled in the example; run `tune` on the target machine before production.

Documents are split into bounded, overlapping character chunks during planning.
When a document receives multiple examples, its slots iterate over chunk IDs in
order before cycling back. Configure this with `target.chunk_size_characters`
and `target.chunk_overlap_characters`.

## Configuration

Configurations are strict JSON. Unknown keys fail validation. Generate an editor-compatible schema with:

```bash
sft-dataset schema --output sft-project.schema.json
```

Secrets are referenced by environment variable names and are never stored in resolved run configurations. The default Hugging Face publication token variable is `HF_TOKEN`.

## Development

```bash
pytest
```

CPU tests use the fake backend. GPU model loading is covered by opt-in smoke tests and should be run on the target cluster before production.
