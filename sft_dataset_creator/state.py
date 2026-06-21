from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterator

from sft_dataset_creator.models import DatasetPlan, EvaluationResult, PlannedSlot, SFTCandidate


SCHEMA = """
CREATE TABLE IF NOT EXISTS slots (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    task TEXT NOT NULL,
    difficulty TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    chunk_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    attempt_count INTEGER NOT NULL DEFAULT 0,
    accepted_candidate TEXT
);
CREATE TABLE IF NOT EXISTS attempts (
    candidate_id TEXT PRIMARY KEY,
    slot_id TEXT NOT NULL,
    attempt_no INTEGER NOT NULL,
    document_id TEXT NOT NULL,
    status TEXT NOT NULL,
    candidate_json TEXT,
    request_json TEXT,
    response_json TEXT,
    evaluation_json TEXT,
    error TEXT,
    request_id TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    latency_seconds REAL,
    speculative INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(slot_id) REFERENCES slots(id)
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


class RunState:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(attempts)")}
        additions = {
            "request_id": "TEXT",
            "input_tokens": "INTEGER",
            "output_tokens": "INTEGER",
            "latency_seconds": "REAL",
            "speculative": "INTEGER NOT NULL DEFAULT 0",
        }
        with self.connection:
            for name, definition in additions.items():
                if name not in columns:
                    self.connection.execute(f"ALTER TABLE attempts ADD COLUMN {name} {definition}")
            slot_columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(slots)")}
            if "chunk_id" not in slot_columns:
                self.connection.execute("ALTER TABLE slots ADD COLUMN chunk_id TEXT")

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "RunState":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def initialize(self, plan: DatasetPlan) -> None:
        with self.connection:
            self.connection.executemany(
                "INSERT OR IGNORE INTO slots(id, document_id, task, difficulty, ordinal, chunk_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (slot.id, slot.document_id, slot.task, slot.difficulty, slot.ordinal, slot.chunk_id)
                    for slot in plan.slots
                ],
            )

    def pending_slots(self, max_attempts: int) -> list[tuple[PlannedSlot, int]]:
        rows = self.connection.execute(
            "SELECT * FROM slots WHERE status != 'accepted' AND attempt_count < ? ORDER BY ordinal",
            (max_attempts,),
        ).fetchall()
        return [
            (
                PlannedSlot(
                    id=row["id"],
                    document_id=row["document_id"],
                    task=row["task"],
                    difficulty=row["difficulty"],
                    ordinal=row["ordinal"],
                    chunk_id=row["chunk_id"],
                ),
                int(row["attempt_count"]) + 1,
            )
            for row in rows
        ]

    def record_attempt(
        self,
        candidate: SFTCandidate | None,
        *,
        slot_id: str,
        attempt: int,
        document_id: str,
        request_json: str | None = None,
        response_json: str | None = None,
        error: str | None = None,
        request_id: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        latency_seconds: float | None = None,
    ) -> None:
        candidate_id = candidate.id if candidate else f"{slot_id}-a{attempt}"
        status = "generated" if candidate else "error"
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO attempts("
                "candidate_id, slot_id, attempt_no, document_id, status, candidate_json, request_json, "
                "response_json, error, request_id, input_tokens, output_tokens, latency_seconds"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    candidate_id,
                    slot_id,
                    attempt,
                    document_id,
                    status,
                    candidate.model_dump_json() if candidate else None,
                    request_json,
                    response_json,
                    error,
                    request_id,
                    input_tokens,
                    output_tokens,
                    latency_seconds,
                ),
            )
            self.connection.execute(
                "UPDATE slots SET attempt_count = MAX(attempt_count, ?) WHERE id = ?",
                (attempt, slot_id),
            )

    def attempt_keys(self) -> set[tuple[str, int]]:
        return {
            (str(row["slot_id"]), int(row["attempt_no"]))
            for row in self.connection.execute("SELECT slot_id, attempt_no FROM attempts")
        }

    def accepted_slot_ids(self) -> set[str]:
        return {
            str(row["id"])
            for row in self.connection.execute("SELECT id FROM slots WHERE status = 'accepted'")
        }

    def generated_candidates(self) -> Iterator[SFTCandidate]:
        rows = self.connection.execute(
            "SELECT candidate_json FROM attempts WHERE status = 'generated' AND candidate_json IS NOT NULL "
            "ORDER BY slot_id, attempt_no"
        )
        for row in rows:
            yield SFTCandidate.model_validate_json(row["candidate_json"])

    def attempted_document_counts(self) -> dict[str, int]:
        return {
            str(row["document_id"]): int(row["count"])
            for row in self.connection.execute(
                "SELECT document_id, COUNT(*) count FROM attempts GROUP BY document_id"
            )
        }

    def mark_superseded(self, candidate: SFTCandidate, reason: str = "slot_already_accepted") -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE attempts SET status = 'superseded', speculative = 1, error = ? WHERE candidate_id = ?",
                (reason, candidate.id),
            )

    def record_evaluation(self, candidate: SFTCandidate, evaluation: EvaluationResult) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE attempts SET status = ?, evaluation_json = ? WHERE candidate_id = ?",
                (evaluation.verdict, evaluation.model_dump_json(), candidate.id),
            )
            if evaluation.verdict == "accept":
                self.connection.execute(
                    "UPDATE slots SET status = 'accepted', accepted_candidate = ? WHERE id = ?",
                    (candidate.model_dump_json(), candidate.slot_id),
                )

    def accepted_candidates(self) -> Iterator[SFTCandidate]:
        rows = self.connection.execute(
            "SELECT accepted_candidate FROM slots WHERE status = 'accepted' ORDER BY ordinal"
        )
        for row in rows:
            yield SFTCandidate.model_validate_json(row["accepted_candidate"])

    def accepted_document_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for candidate in self.accepted_candidates():
            counts[candidate.document_id] = counts.get(candidate.document_id, 0) + 1
        return counts

    def counts(self) -> dict[str, int]:
        slots = self.connection.execute(
            "SELECT COUNT(*) total, SUM(status = 'accepted') accepted FROM slots"
        ).fetchone()
        attempts = self.connection.execute(
            "SELECT COUNT(*) total, SUM(status = 'reject') rejected, SUM(status = 'review') reviewed, "
            "SUM(speculative = 1) speculative FROM attempts"
        ).fetchone()
        llm_judged = self.connection.execute(
            "SELECT COUNT(*) FROM attempts WHERE evaluation_json LIKE '%\"selected_for_llm\":true%'"
        ).fetchone()[0]
        return {
            "target": int(slots["total"] or 0),
            "accepted": int(slots["accepted"] or 0),
            "attempted": int(attempts["total"] or 0),
            "rejected": int(attempts["rejected"] or 0),
            "reviewed": int(attempts["reviewed"] or 0),
            "speculative": int(attempts["speculative"] or 0),
            "llm_judged": int(llm_judged or 0),
        }

    def token_totals(self) -> dict[str, int]:
        row = self.connection.execute(
            "SELECT SUM(input_tokens) input_tokens, SUM(output_tokens) output_tokens FROM attempts"
        ).fetchone()
        return {
            "input_tokens": int(row["input_tokens"] or 0),
            "output_tokens": int(row["output_tokens"] or 0),
        }

    def deficits(self) -> dict[str, int]:
        rows = self.connection.execute(
            "SELECT task, difficulty, COUNT(*) count FROM slots WHERE status != 'accepted' GROUP BY task, difficulty"
        ).fetchall()
        return {f"{row['task']}:{row['difficulty']}": int(row["count"]) for row in rows}

    def event(self, kind: str, payload: dict) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO events(kind, payload) VALUES (?, ?)",
                (kind, json.dumps(payload, ensure_ascii=False)),
            )
