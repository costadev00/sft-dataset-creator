from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sft_dataset_creator.chunking import chunk_document
from sft_dataset_creator.config import DistributionConfig, ProjectConfig, save_config
from sft_dataset_creator.models import DatasetPlan, Document, DocumentIndex, PlannedSlot, RunManifest
from sft_dataset_creator.profiles import document_is_eligible
from sft_dataset_creator.registry import create
from sft_dataset_creator.sources import get_nested


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


def _snapshot_documents(config: ProjectConfig, selected_ids: set[str], target: Path) -> None:
    source = create("sources", config.source.plugin, config.source)
    found: set[str] = set()
    with target.open("w", encoding="utf-8") as handle:
        for document in source.iter_documents():
            if document.id not in selected_ids:
                continue
            chunked = chunk_document(
                document,
                config.target.chunk_size_characters,
                config.target.chunk_overlap_characters,
            )
            handle.write(chunked.model_dump_json() + "\n")
            found.add(document.id)
    missing = selected_ids - found
    if missing:
        raise RuntimeError(f"source was not replayable; missing selected document ids: {sorted(missing)[:5]}")


def build_plan(config: ProjectConfig, run_dir: str | Path | None = None) -> DatasetPlan:
    run_id = _run_id(config.name, config.config_hash)
    target_dir = Path(run_dir) if run_dir else config.runtime.run_root / run_id
    target_dir.mkdir(parents=True, exist_ok=False)
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
    tasks = _expand_counts(_distribution_counts(config.composition.tasks, config.target.examples), config.selection.seed)
    difficulties = _expand_counts(
        _distribution_counts(config.composition.difficulties, config.target.examples),
        config.selection.seed + 1,
    )
    primary = sorted(primary, key=lambda item: _stable_value(config.selection.seed + 2, item.id))
    minimum = config.target.per_document.minimum
    if minimum > config.target.examples:
        shutil.rmtree(target_dir)
        raise ValueError("per-document minimum exceeds the final example target")
    active_count = min(len(primary), config.target.examples if minimum == 0 else config.target.examples // minimum)
    active = primary[:active_count]
    document_loads = {item.id: 0 for item in active}
    document_order: list[DocumentIndex] = []
    if minimum:
        for item in active:
            document_order.extend([item] * minimum)
    provisional = Counter(item.id for item in document_order)
    while len(document_order) < config.target.examples:
        candidates = [item for item in active if provisional[item.id] < config.target.per_document.maximum]
        if not candidates:
            shutil.rmtree(target_dir)
            raise ValueError("per-document constraints cannot satisfy the requested target")
        candidates.sort(key=lambda item: (provisional[item.id], _stable_value(config.selection.seed + len(document_order), item.id)))
        document_order.append(candidates[0])
        provisional[candidates[0].id] += 1
    random.Random(config.selection.seed + 3).shuffle(document_order)
    slots: list[PlannedSlot] = []
    for ordinal in range(config.target.examples):
        document = document_order[ordinal]
        chunk_count = int(document.metadata.get("chunk_count", 1))
        chunk_id = str(document_loads[document.id] % chunk_count)
        document_loads[document.id] += 1
        slots.append(
            PlannedSlot(
                id=f"slot-{ordinal + 1:08d}",
                document_id=document.id,
                task=tasks[ordinal],
                difficulty=difficulties[ordinal],
                ordinal=ordinal + 1,
                chunk_id=chunk_id,
            )
        )
    snapshot_path = target_dir / "corpus-selected.jsonl"
    selected_ids = {item.id for item in selected}
    _snapshot_documents(config, selected_ids, snapshot_path)
    plan = DatasetPlan(
        project_name=config.name,
        run_id=run_id,
        config_hash=config.config_hash,
        seed=config.selection.seed,
        corpus_snapshot=str(snapshot_path.resolve()),
        documents=selected,
        reserve_document_ids=[item.id for item in reserve],
        slots=slots,
        estimates={
            "eligible_documents": len(indexes),
            "selected_documents": len(selected),
            "primary_documents": len(active),
            "reserve_documents": len(reserve),
            "planned_examples": len(slots),
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
        database_path=str((target_dir / "run.db").resolve()),
    )
    (target_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return plan


def load_plan(path: str | Path) -> DatasetPlan:
    return DatasetPlan.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_document_snapshot(path: str | Path) -> dict[str, Document]:
    documents: dict[str, Document] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                document = Document.model_validate_json(line)
                documents[document.id] = document
    return documents
