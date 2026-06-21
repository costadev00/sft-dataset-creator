# Architecture

## Public contracts

The public Python surface consists of `ProjectConfig`, `BatchingConfig`, `BatchGenerationResult`, `load_config`, `build_plan`, `execute_plan`, `tune_project`, `export_run`, and `publish_run`. Core records use Pydantic and reject unknown fields.

Extension points are Python Protocols discovered through these entry-point groups:

- `sft_dataset_creator.sources`
- `sft_dataset_creator.tasks`
- `sft_dataset_creator.backends`
- `sft_dataset_creator.evaluators`
- `sft_dataset_creator.exporters`

A source emits canonical `Document` records. A task recipe converts a planned slot and document into a `GenerationRequest`. A backend returns structured JSON. Backends may additionally implement batch generation and vectorized token counting; the engine detects those capabilities and falls back to the synchronous contract. Evaluators operate on canonical `SFTCandidate` records. Exporters are views over accepted candidates rather than separate generation pipelines.

## Planning

Planning scans the source without calling a model. It validates unique document IDs, applies profile eligibility and configured filters, records a lightweight index, and performs seeded stratified sampling. Selected documents are split into deterministic, bounded chunks with stable numeric section IDs. Multiple slots assigned to one document iterate over those chunks in order.

Task and difficulty weights are normalized and converted to integer quotas with largest-remainder apportionment. Slots are distributed across documents by current load, giving coverage priority before reuse. The configured reserve remains unassigned until replacement is necessary.

`plan.json` is immutable and tied to `config.resolved.json` by SHA-256. The selected document bodies are stored separately in `corpus-selected.jsonl` so execution does not depend on the source remaining unchanged.

## Execution and recovery

`run.db` is the transactional source of truth for slots, attempts, evaluations, and accepted canonical examples. An interrupted run resumes slots whose status is not accepted and never regenerates completed work.

Execution plans attempts round-robin, capped by both the per-slot and global budgets. The generator is loaded once in an isolated spawned process. A bounded producer prepares requests in vectorized tokenizer batches while `AsyncLLM` continuously schedules GPU work. Results may complete in any order, but SQLite commits and quality decisions follow stable slot and attempt order.

Each generation request receives one planned chunk rather than the complete document. Evidence offsets are local to that chunk and remain verifiable because the chunked corpus is stored in the immutable snapshot.

Deterministic gates validate evidence and self-contained SFT content. Duplicate detection is intentionally left to downstream processing. References to hidden source material such as "according to the text" or "cited in the document" are rejected across instruction, input, and output. The exporter repeats this gate so records accepted by older versions cannot be published. Routed candidates enter one continuous batch in a second isolated model process. Only the first accepted attempt can satisfy a slot; later generated attempts are marked `superseded` and retain their token cost as speculative work. Synchronous third-party backends and evaluators continue to run sequentially.

The application runs as local Python processes; Docker is not part of the runtime architecture. Production model prompts are centralized in `sft_dataset_creator/prompts.py`.

The first attempts use the planned document. Later attempts use reserve documents after the configured same-document retry count. Runs that exhaust the global or per-slot attempt budgets are marked `partial` and retain usable exports plus an explicit deficit report. Generated but unevaluated attempts are resumed without another model call.

## Audit artifacts

Each run directory contains:

```text
config.resolved.json
manifest.json
plan.json
corpus-index.jsonl
corpus-selected.jsonl
run.db
report.json
exports/
```

The manifest records hardware, installed plugins, package availability, models, and configuration hashes. Final exports retain document IDs, task metadata, evidence spans, and model provenance. Hidden reasoning and secrets are never persisted.

`audit-sample` creates a blinded sample balanced across task, difficulty, and LLM-routing status. The system decisions are written to a separate key. After human labels are added, `audit-score` reports human acceptance, selective-routing rate, and recall over human-rejected candidates.
