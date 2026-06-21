from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from sft_dataset_creator.config import OutputConfig, ProjectConfig, load_config
from sft_dataset_creator.models import SFTCandidate
from sft_dataset_creator.quality import candidate_source_reference
from sft_dataset_creator.registry import create, register
from sft_dataset_creator.state import RunState


class MessagesExporter:
    name = "messages"

    def render(self, candidate: SFTCandidate) -> dict[str, Any]:
        return {
            "id": candidate.id,
            "messages": [message.model_dump() for message in candidate.messages],
            **_metadata(candidate),
        }


class PromptCompletionExporter:
    name = "prompt_completion"

    def render(self, candidate: SFTCandidate) -> dict[str, Any]:
        prompt = candidate.instruction
        if candidate.input:
            prompt = f"{prompt}\n\nContext:\n{candidate.input}"
        return {"id": candidate.id, "prompt": prompt, "completion": candidate.output, **_metadata(candidate)}


class AlpacaExporter:
    name = "alpaca"

    def render(self, candidate: SFTCandidate) -> dict[str, Any]:
        return {
            "id": candidate.id,
            "instruction": candidate.instruction,
            "input": candidate.input,
            "output": candidate.output,
            **_metadata(candidate),
        }


def _metadata(candidate: SFTCandidate) -> dict[str, Any]:
    return {
        "source": candidate.source,
        "document_id": candidate.document_id,
        "source_title": candidate.source_title,
        "task": candidate.task,
        "difficulty": candidate.difficulty,
        "evidence": [item.model_dump() for item in candidate.evidence],
        "generator": candidate.generator,
        "model": candidate.model,
    }


def _split_assignments(candidates: list[SFTCandidate], output: OutputConfig, seed: int) -> dict[str, str]:
    by_document: dict[str, list[SFTCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_document[candidate.document_id].append(candidate)
    strata: dict[tuple[tuple[str, ...], tuple[str, ...]], list[str]] = defaultdict(list)
    for document_id, items in by_document.items():
        task_counts = defaultdict(int)
        difficulty_counts = defaultdict(int)
        for item in items:
            task_counts[item.task] += 1
            difficulty_counts[item.difficulty] += 1
        primary_task = min(task_counts, key=lambda item: (-task_counts[item], item))
        primary_difficulty = min(difficulty_counts, key=lambda item: (-difficulty_counts[item], item))
        key = ((primary_task,), (primary_difficulty,))
        strata[key].append(document_id)
    assignments: dict[str, str] = {}
    ratios = {"train": output.splits.train, "validation": output.splits.validation, "test": output.splits.test}
    for key, document_ids in sorted(strata.items()):
        document_ids.sort(key=lambda item: hashlib.sha256(f"{seed}:{item}".encode()).digest())
        raw = {name: len(document_ids) * ratio for name, ratio in ratios.items()}
        counts = {name: math.floor(value) for name, value in raw.items()}
        for name in sorted(raw, key=lambda item: (raw[item] - counts[item], item), reverse=True)[: len(document_ids) - sum(counts.values())]:
            counts[name] += 1
        position = 0
        for split in ("train", "validation", "test"):
            for document_id in document_ids[position : position + counts[split]]:
                assignments[document_id] = split
            position += counts[split]
    return assignments


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Parquet exports require the 'hf' extra") from exc
    pq.write_table(pa.Table.from_pylist(rows), path)


def _dataset_card(config: ProjectConfig, examples: int, enabled_splits: set[str]) -> str:
    container = "parquet" if "parquet" in config.output.containers else "jsonl"
    lines = ["---", "configs:"]
    for format_name in config.output.formats:
        lines.append(f"- config_name: {format_name}")
        lines.append("  data_files:")
        for split in ("train", "validation", "test"):
            if split in enabled_splits:
                lines.extend(
                    [
                        f"  - split: {split}",
                        f"    path: {format_name}/{split}.{container}",
                    ]
                )
    lines.extend(
        [
            "---",
            "",
            f"# {config.name}",
            "",
            "Synthetic SFT dataset generated by `sft-dataset-creator`.",
            "",
            f"- Examples: {examples}",
            f"- Generator: `{config.generation.model}`",
            f"- Evaluator: `{config.evaluation.llm.model if config.evaluation.llm else 'deterministic only'}`",
            f"- Configuration hash: `{config.config_hash}`",
            "",
            "Review provenance, licenses, `report.json`, and the resolved configuration before public release.",
            "",
        ]
    )
    return "\n".join(lines)


def export_run(run_dir: str | Path, *, config: ProjectConfig | None = None) -> Path:
    root = Path(run_dir)
    if config is None:
        config = load_config(root / "config.resolved.json")
    with RunState(root / "run.db") as state:
        candidates = list(state.accepted_candidates())
    source_dependent = [
        candidate.id
        for candidate in candidates
        if candidate_source_reference(candidate.instruction, candidate.input, candidate.output)
    ]
    if source_dependent:
        preview = ", ".join(source_dependent[:10])
        remainder = len(source_dependent) - min(len(source_dependent), 10)
        suffix = f" and {remainder} more" if remainder else ""
        raise ValueError(
            "refusing to export source-dependent SFT content; "
            f"re-evaluate or regenerate candidates: {preview}{suffix}"
        )
    export_root = root / "exports"
    export_root.mkdir(parents=True, exist_ok=True)
    assignments = _split_assignments(candidates, config.output, config.selection.seed)
    split_ratios = {
        "train": config.output.splits.train,
        "validation": config.output.splits.validation,
        "test": config.output.splits.test,
    }
    enabled_splits = {name for name, ratio in split_ratios.items() if ratio > 0.0}
    for format_name in config.output.formats:
        exporter = create("exporters", format_name, None)
        format_dir = export_root / format_name
        format_dir.mkdir(parents=True, exist_ok=True)
        by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
        for candidate in candidates:
            by_split[assignments[candidate.document_id]].append(exporter.render(candidate))
        for split, rows in by_split.items():
            if split not in enabled_splits:
                for container in config.output.containers:
                    stale_path = format_dir / f"{split}.{container}"
                    stale_path.unlink(missing_ok=True)
                continue
            if "jsonl" in config.output.containers:
                _write_jsonl(format_dir / f"{split}.jsonl", rows)
            if "parquet" in config.output.containers:
                _write_parquet(format_dir / f"{split}.parquet", rows)
    info = {
        "project": config.name,
        "examples": len(candidates),
        "formats": config.output.formats,
        "containers": config.output.containers,
        "generator": config.generation.model,
        "evaluator": config.evaluation.llm.model if config.evaluation.llm else None,
        "config_hash": config.config_hash,
    }
    (export_root / "dataset_info.json").unlink(missing_ok=True)
    (export_root / "generation_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (export_root / "README.md").write_text(
        _dataset_card(config, len(candidates), enabled_splits),
        encoding="utf-8",
    )
    return export_root


register("exporters", "messages", lambda _config: MessagesExporter())
register("exporters", "prompt_completion", lambda _config: PromptCompletionExporter())
register("exporters", "alpaca", lambda _config: AlpacaExporter())
