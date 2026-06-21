from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from sft_dataset_creator.models import EvaluationResult, SFTCandidate


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def create_audit_sample(run_dir: str | Path, *, size: int = 300, seed: int = 42) -> Path:
    root = Path(run_dir)
    connection = sqlite3.connect(root / "run.db")
    rows = connection.execute(
        "SELECT candidate_json, evaluation_json FROM attempts WHERE candidate_json IS NOT NULL AND evaluation_json IS NOT NULL"
    ).fetchall()
    connection.close()
    groups: dict[tuple[str, str, bool], list[tuple[SFTCandidate, EvaluationResult]]] = defaultdict(list)
    for candidate_json, evaluation_json in rows:
        candidate = SFTCandidate.model_validate_json(candidate_json)
        evaluation = EvaluationResult.model_validate_json(evaluation_json)
        groups[(candidate.task, candidate.difficulty, evaluation.selected_for_llm)].append((candidate, evaluation))
    for items in groups.values():
        items.sort(key=lambda item: hashlib.sha256(f"{seed}:{item[0].id}".encode("utf-8")).digest())
    selected: list[tuple[SFTCandidate, EvaluationResult]] = []
    active = sorted(groups)
    while active and len(selected) < min(size, len(rows)):
        next_active = []
        for key in active:
            if groups[key] and len(selected) < size:
                selected.append(groups[key].pop(0))
            if groups[key]:
                next_active.append(key)
        active = next_active
    audit_dir = root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    review_rows = []
    key_rows = []
    for index, (candidate, evaluation) in enumerate(selected, start=1):
        audit_id = f"audit-{index:04d}"
        review_rows.append(
            {
                "audit_id": audit_id,
                "task": candidate.task,
                "difficulty": candidate.difficulty,
                "instruction": candidate.instruction,
                "input": candidate.input,
                "output": candidate.output,
                "evidence": [item.model_dump() for item in candidate.evidence],
                "source_title": candidate.source_title,
                "human_verdict": None,
                "human_issues": [],
            }
        )
        key_rows.append(
            {
                "audit_id": audit_id,
                "candidate_id": candidate.id,
                "system_verdict": evaluation.verdict,
                "selected_for_llm": evaluation.selected_for_llm,
                "evaluator": evaluation.evaluator,
                "system_issues": evaluation.issues,
            }
        )
    review_path = audit_dir / "review.jsonl"
    _write_jsonl(review_path, review_rows)
    _write_jsonl(audit_dir / "key.jsonl", key_rows)
    (audit_dir / "README.md").write_text(
        "# Blind audit\n\nFill `human_verdict` with `accept`, `reject`, or `review` and add optional "
        "`human_issues`. Do not open `key.jsonl` until review is complete. Then run "
        "`sft-dataset audit-score <run-dir>`.\n",
        encoding="utf-8",
    )
    return review_path


def score_audit(run_dir: str | Path) -> dict[str, Any]:
    audit_dir = Path(run_dir) / "audit"
    review = {row["audit_id"]: row for row in _read_jsonl(audit_dir / "review.jsonl")}
    key = {row["audit_id"]: row for row in _read_jsonl(audit_dir / "key.jsonl")}
    missing = [
        audit_id
        for audit_id, row in review.items()
        if row.get("human_verdict") not in {"accept", "reject", "review"}
    ]
    if missing:
        raise ValueError(f"audit is incomplete; {len(missing)} rows have no valid human_verdict")
    total = len(review)
    human_rejected = [audit_id for audit_id, row in review.items() if row["human_verdict"] != "accept"]
    caught = [audit_id for audit_id in human_rejected if key[audit_id]["system_verdict"] != "accept"]
    routed = sum(1 for value in key.values() if value["selected_for_llm"])
    reject_recall = len(caught) / len(human_rejected) if human_rejected else 1.0
    report = {
        "sample_size": total,
        "human_accept_rate": (
            sum(row["human_verdict"] == "accept" for row in review.values()) / total if total else 0.0
        ),
        "human_rejected": len(human_rejected),
        "caught_human_rejections": len(caught),
        "reject_recall": reject_recall,
        "llm_route_rate": routed / total if total else 0.0,
        "meets_reject_recall_target": reject_recall >= 0.90,
    }
    (audit_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
