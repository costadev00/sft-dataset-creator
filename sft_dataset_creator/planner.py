from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from sft_dataset_creator.chunking import chunk_document
from sft_dataset_creator.config import DistributionConfig, ProjectConfig, save_config
from sft_dataset_creator.models import DatasetPlan, Document, DocumentIndex, PlannedSlot, RunManifest
from sft_dataset_creator.profiles import document_is_eligible
from sft_dataset_creator.registry import create
from sft_dataset_creator.sources import get_nested
from sft_dataset_creator.state import RunState


def _run_id(name: str, config_hash: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = "".join(char.lower() if char.isalnum() else "-" for char in name).strip("-")
    return f"{timestamp}-{safe_name}-{config_hash[:8]}"


def _stable_value(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _matches_filters(document: Document, filters: dict[str, Any]) -> bool:
    container = {"id": document.id, "title": document.title, "license": document.license, "metadata": document.metadata}
    for path, expected in filters.items():
        actual = get_nested(container, path)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _stratum(document: Document, paths: list[str]) -> list[str]:
    container = {"id": document.id, "title": document.title, "license": document.license, "metadata": document.metadata}
    return [str(get_nested(container, path, "<missing>")) for path in paths]


def _apportion(total: int, weights: dict[str, float]) -> dict[str, int]:
    weight_total = sum(weights.values())
    raw = {key: total * value / weight_total for key, value in weights.items()}
    allocated = {key: math.floor(value) for key, value in raw.items()}
    remaining = total - sum(allocated.values())
    order = sorted(raw, key=lambda key: (raw[key] - allocated[key], key), reverse=True)
    for key in order[:remaining]:
        allocated[key] += 1
    return allocated


def _distribution_counts(spec: DistributionConfig, total: int) -> dict[str, int]:
    if spec.counts is not None:
        if sum(spec.counts.values()) != total:
            raise ValueError(f"exact distribution counts must sum to target {total}")
        return dict(spec.counts)
    return _apportion(total, spec.weights or {})


def _expand_counts(counts: dict[str, int], seed: int) -> list[str]:
    values = [name for name, count in sorted(counts.items()) for _ in range(count)]
    random.Random(seed).shuffle(values)
    return values


def _select_documents(indexes: list[DocumentIndex], config: ProjectConfig) -> list[DocumentIndex]:
    if config.selection.count is not None:
        wanted = min(config.selection.count, len(indexes))
    else:
        wanted = max(1, math.ceil(len(indexes) * float(config.selection.fraction)))
    groups: dict[tuple[str, ...], list[DocumentIndex]] = defaultdict(list)
    for item in indexes:
        groups[tuple(item.stratum)].append(item)
    group_weights = {"\x1f".join(key): len(value) for key, value in groups.items()}
    group_counts = _apportion(wanted, group_weights)
    selected: list[DocumentIndex] = []
    for key, items in sorted(groups.items()):
        items.sort(key=lambda item: _stable_value(config.selection.seed, item.id))
        selected.extend(items[: group_counts.get("\x1f".join(key), 0)])
    if len(selected) < wanted:
        selected_ids = {item.id for item in selected}
        remaining = sorted(
            (item for item in indexes if item.id not in selected_ids),
            key=lambda item: _stable_value(config.selection.seed, item.id),
        )
        selected.extend(remaining[: wanted - len(selected)])
    return selected


def _scheduled_value(counts: dict[str, int], assigned: dict[str, int], ordinal: int, seed: int) -> str:
    candidates = [name for name, count in counts.items() if assigned.get(name, 0) < count]
    if not candidates:
        raise ValueError("distribution is exhausted before all slots were assigned")
    candidates.sort(
        key=lambda name: (
            assigned.get(name, 0) / counts[name],
            _stable_value(seed + ordinal, name),
            name,
        )
    )
    selected = candidates[0]
    assigned[selected] = assigned.get(selected, 0) + 1
    return selected


def _iter_document_order(
    active: list[DocumentIndex],
    total: int,
    minimum: int,
    maximum: int,
    seed: int,
) -> Iterator[DocumentIndex]:
    assigned = {item.id: 0 for item in active}
    emitted = 0
    for pass_index in range(minimum):
        ordered = sorted(active, key=lambda item: (_stable_value(seed + pass_index, item.id), item.id))
        for item in ordered:
            if emitted >= total:
                return
            if assigned[item.id] >= maximum:
                continue
            assigned[item.id] += 1
            emitted += 1
            yield item
    while emitted < total:
        candidates = [item for item in active if assigned[item.id] < maximum]
        if not candidates:
            raise ValueError("per-document constraints cannot satisfy the requested target")
        candidates.sort(
            key=lambda item: (
                assigned[item.id],
                _stable_value(seed + 1_000_003 + emitted, item.id),
                item.id,
            )
        )
        selected = candidates[0]
        assigned[selected.id] += 1
        emitted += 1
        yield selected


def _snapshot_documents(
    config: ProjectConfig,
    selected_ids: set[str],
    roles: dict[str, str],
    target_dir: Path,
    state: RunState,
    shard_size: int = 10_000,
) -> Path:
    source = create("sources", config.source.plugin, config.source)
    found: set[str] = set()
    snapshot_dir = target_dir / "corpus-selected"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    shard_index = 0
    shard_count = 0
    shard_bytes = 0
    handle = None
    shard_path: Path | None = None
    ordinal = 0
    document_rows: list[dict[str, Any]] = []
    try:
        for document in source.iter_documents():
            if document.id not in selected_ids:
                continue
            if handle is None or shard_count >= shard_size:
                if handle is not None and shard_path is not None:
                    handle.close()
                    state.record_document_shard(str(shard_path.relative_to(target_dir)), shard_count, shard_bytes)
                shard_path = snapshot_dir / f"documents-{shard_index:06d}.jsonl"
                handle = shard_path.open("wb")
                shard_index += 1
                shard_count = 0
                shard_bytes = 0
            chunked = chunk_document(
                document,
                config.target.chunk_size_characters,
                config.target.chunk_overlap_characters,
            )
            payload = (chunked.model_dump_json() + "\n").encode("utf-8")
            offset = handle.tell()
            handle.write(payload)
            document_rows.append(
                {
                    "document_id": chunked.id,
                    "source": chunked.source,
                    "title": chunked.title,
                    "estimated_tokens": _estimate_tokens(chunked.text),
                    "stratum": _stratum(chunked, config.selection.strata),
                    "metadata": chunked.metadata,
                    "role": roles.get(chunked.id, "primary"),
                    "shard_path": str(shard_path.relative_to(target_dir)),
                    "byte_offset": offset,
                    "byte_length": len(payload),
                    "ordinal": ordinal,
                }
            )
            if len(document_rows) >= 1_000:
                state.record_documents(document_rows)
                document_rows = []
            found.add(document.id)
            ordinal += 1
            shard_count += 1
            shard_bytes += len(payload)
    finally:
        if document_rows:
            state.record_documents(document_rows)
        if handle is not None:
            handle.close()
            if shard_path is not None:
                state.record_document_shard(str(shard_path.relative_to(target_dir)), shard_count, shard_bytes)
    missing = selected_ids - found
    if missing:
        raise RuntimeError(f"source was not replayable; missing selected document ids: {sorted(missing)[:5]}")
    return snapshot_dir


def build_plan(config: ProjectConfig, run_dir: str | Path | None = None) -> DatasetPlan:
    run_id = _run_id(config.name, config.config_hash)
    target_dir = Path(run_dir) if run_dir else config.runtime.run_root / run_id
    target_dir.mkdir(parents=True, exist_ok=False)
    database_path = target_dir / "run.db"
    source = create("sources", config.source.plugin, config.source)
    indexes: list[DocumentIndex] = []
    seen_ids: set[str] = set()
    scan_path = target_dir / "corpus-index.jsonl"
    with scan_path.open("w", encoding="utf-8") as handle:
        for document in source.iter_documents():
            if document.id in seen_ids:
                raise ValueError(f"duplicate document id from source: {document.id}")
            seen_ids.add(document.id)
            if not document_is_eligible(document, config.profile) or not _matches_filters(document, config.selection.filters):
                continue
            chunked = chunk_document(
                document,
                config.target.chunk_size_characters,
                config.target.chunk_overlap_characters,
            )
            chunk_tokens = [_estimate_tokens(section.text) for section in chunked.sections]
            index = DocumentIndex(
                id=document.id,
                source=document.source,
                title=document.title,
                estimated_tokens=_estimate_tokens(document.text),
                stratum=_stratum(document, config.selection.strata),
                metadata={
                    "license": document.license,
                    "chunk_count": len(chunked.sections),
                    "average_chunk_tokens": math.ceil(sum(chunk_tokens) / len(chunk_tokens)),
                },
            )
            indexes.append(index)
            handle.write(index.model_dump_json() + "\n")
    if not indexes:
        shutil.rmtree(target_dir)
        raise ValueError("no eligible documents found in source")
    selected = _select_documents(indexes, config)
    selected.sort(key=lambda item: _stable_value(config.selection.seed + 4, item.id))
    reserve_count = min(math.ceil(len(selected) * config.target.reserve_fraction), max(0, len(selected) - 1))
    primary = selected[:-reserve_count] if reserve_count else selected
    reserve = selected[-reserve_count:] if reserve_count else []
    if len(primary) * config.target.per_document.maximum < config.target.examples:
        shutil.rmtree(target_dir)
        raise ValueError(
            "selected primary corpus cannot satisfy target with the configured per-document maximum; "
            "increase selection size or reduce reserve_fraction"
        )
    task_counts = _distribution_counts(config.composition.tasks, config.target.examples)
    difficulty_counts = _distribution_counts(config.composition.difficulties, config.target.examples)
    assigned_tasks: dict[str, int] = {}
    assigned_difficulties: dict[str, int] = {}
    primary = sorted(primary, key=lambda item: _stable_value(config.selection.seed + 2, item.id))
    minimum = config.target.per_document.minimum
    if minimum > config.target.examples:
        shutil.rmtree(target_dir)
        raise ValueError("per-document minimum exceeds the final example target")
    active_count = min(len(primary), config.target.examples if minimum == 0 else config.target.examples // minimum)
    active = primary[:active_count]
    document_loads = {item.id: 0 for item in active}
    roles = {item.id: "primary" for item in primary}
    roles.update({item.id: "reserve" for item in reserve})
    selected_ids = {item.id for item in selected}
    with RunState(database_path) as state:
        snapshot_path = _snapshot_documents(config, selected_ids, roles, target_dir, state)
        slot_batch: list[PlannedSlot] = []
        try:
            document_order = _iter_document_order(
                active,
                config.target.examples,
                minimum,
                config.target.per_document.maximum,
                config.selection.seed + 3,
            )
            for ordinal, document in enumerate(document_order):
                chunk_count = int(document.metadata.get("chunk_count", 1))
                chunk_id = str(document_loads[document.id] % chunk_count)
                document_loads[document.id] += 1
                slot_batch.append(
                    PlannedSlot(
                        id=f"slot-{ordinal + 1:08d}",
                        document_id=document.id,
                        task=_scheduled_value(task_counts, assigned_tasks, ordinal, config.selection.seed),
                        difficulty=_scheduled_value(
                            difficulty_counts,
                            assigned_difficulties,
                            ordinal,
                            config.selection.seed + 1,
                        ),
                        ordinal=ordinal + 1,
                        chunk_id=chunk_id,
                    )
                )
                if len(slot_batch) >= 10_000:
                    state.insert_slots(slot_batch)
                    slot_batch = []
        except ValueError:
            shutil.rmtree(target_dir)
            raise
        if sum(document_loads.values()) != config.target.examples:
            shutil.rmtree(target_dir)
            raise ValueError("planner did not emit the requested number of slots")
        if slot_batch:
            state.insert_slots(slot_batch)
        state.set_metadata(
            "plan",
            {
                "run_id": run_id,
                "project_name": config.name,
                "config_hash": config.config_hash,
                "seed": config.selection.seed,
                "target_examples": config.target.examples,
            },
        )
    plan = DatasetPlan(
        version="2",
        project_name=config.name,
        run_id=run_id,
        config_hash=config.config_hash,
        seed=config.selection.seed,
        corpus_snapshot=str(snapshot_path.resolve()),
        database_path=str(database_path.resolve()),
        documents=[],
        reserve_document_ids=[],
        slots=[],
        estimates={
            "eligible_documents": len(indexes),
            "selected_documents": len(selected),
            "primary_documents": len(active),
            "reserve_documents": len(reserve),
            "planned_examples": config.target.examples,
            "estimated_input_tokens": sum(
                int(item.metadata.get("average_chunk_tokens", item.estimated_tokens))
                * document_loads.get(item.id, 0)
                for item in active
            ),
        },
    )
    config_path = save_config(config, target_dir / "config.resolved.json")
    plan_path = target_dir / "plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    manifest = RunManifest(
        run_id=run_id,
        project_name=config.name,
        status="planned",
        config_path=str(config_path.resolve()),
        plan_path=str(plan_path.resolve()),
        database_path=str(database_path.resolve()),
    )
    (target_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return plan


def load_plan(path: str | Path) -> DatasetPlan:
    return DatasetPlan.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_document_snapshot(path: str | Path) -> dict[str, Document]:
    documents: dict[str, Document] = {}
    target = Path(path)
    paths = sorted(target.glob("*.jsonl")) if target.is_dir() else [target]
    for item in paths:
        with item.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    document = Document.model_validate_json(line)
                    documents[document.id] = document
    return documents
