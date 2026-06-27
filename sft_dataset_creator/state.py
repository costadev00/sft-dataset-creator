from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterator

from sft_dataset_creator.models import DatasetPlan, EvaluationResult, PlannedSlot, SFTCandidate
from sft_dataset_creator.quality import candidate_fingerprint


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
CREATE TABLE IF NOT EXISTS document_shards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    document_count INTEGER NOT NULL DEFAULT 0,
    byte_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    title TEXT,
    estimated_tokens INTEGER NOT NULL,
    stratum_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    role TEXT NOT NULL DEFAULT 'primary',
    shard_path TEXT NOT NULL,
    byte_offset INTEGER NOT NULL,
    byte_length INTEGER NOT NULL,
    ordinal INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS checkpoint_shards (
    format TEXT NOT NULL,
    split TEXT NOT NULL,
    shard_index INTEGER NOT NULL,
    path TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'closed',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    closed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(format, split, shard_index)
);
CREATE TABLE IF NOT EXISTS checkpoint_items (
    candidate_id TEXT NOT NULL,
    format TEXT NOT NULL,
    split TEXT NOT NULL,
    shard_path TEXT NOT NULL,
    row_index INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(candidate_id, format, split)
);
CREATE TABLE IF NOT EXISTS accepted_fingerprints (
    fingerprint TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS run_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


class RunState:
    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        if read_only:
            uri = f"file:{self.path.resolve()}?mode=ro"
            self.connection = sqlite3.connect(uri, uri=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA busy_timeout = 30000")
        if not read_only:
            self.connection.execute("PRAGMA journal_mode = WAL")
            self.connection.executescript(SCHEMA)
            self._migrate()
            self._create_indexes()

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
            document_columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(documents)")}
            if document_columns and "role" not in document_columns:
                self.connection.execute("ALTER TABLE documents ADD COLUMN role TEXT NOT NULL DEFAULT 'primary'")

    def _create_indexes(self) -> None:
        with self.connection:
            self.connection.execute("CREATE INDEX IF NOT EXISTS idx_slots_status_ordinal ON slots(status, ordinal)")
            self.connection.execute("CREATE INDEX IF NOT EXISTS idx_slots_attempt_count ON slots(attempt_count)")
            self.connection.execute("CREATE INDEX IF NOT EXISTS idx_attempts_status ON attempts(status)")
            self.connection.execute("CREATE INDEX IF NOT EXISTS idx_attempts_slot_attempt ON attempts(slot_id, attempt_no)")
            self.connection.execute("CREATE INDEX IF NOT EXISTS idx_documents_role_ordinal ON documents(role, ordinal)")
            self.connection.execute("CREATE INDEX IF NOT EXISTS idx_checkpoint_items_format ON checkpoint_items(format, split)")

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "RunState":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def initialize(self, plan: DatasetPlan) -> None:
        if not plan.slots:
            return
        self.insert_slots(plan.slots)

    def insert_slots(self, slots: list[PlannedSlot]) -> None:
        with self.connection:
            self.connection.executemany(
                "INSERT OR IGNORE INTO slots(id, document_id, task, difficulty, ordinal, chunk_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (slot.id, slot.document_id, slot.task, slot.difficulty, slot.ordinal, slot.chunk_id)
                    for slot in slots
                ],
            )

    def pending_slots(self, max_attempts: int, limit: int | None = None) -> list[tuple[PlannedSlot, int]]:
        suffix = " ORDER BY ordinal"
        params: list[int] = [max_attempts]
        if limit is not None:
            suffix += " LIMIT ?"
            params.append(limit)
        rows = self.connection.execute(
            "SELECT * FROM slots WHERE status != 'accepted' AND attempt_count < ?" + suffix,
            params,
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

    def slots(self, limit: int | None = None) -> list[PlannedSlot]:
        suffix = " ORDER BY ordinal"
        params: list[int] = []
        if limit is not None:
            suffix += " LIMIT ?"
            params.append(limit)
        rows = self.connection.execute("SELECT * FROM slots" + suffix, params).fetchall()
        return [
            PlannedSlot(
                id=row["id"],
                document_id=row["document_id"],
                task=row["task"],
                difficulty=row["difficulty"],
                ordinal=row["ordinal"],
                chunk_id=row["chunk_id"],
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

    def accepted_slot_ids_for(self, slot_ids: set[str]) -> set[str]:
        if not slot_ids:
            return set()
        accepted: set[str] = set()
        values = list(slot_ids)
        for start in range(0, len(values), 900):
            chunk = values[start : start + 900]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                f"SELECT id FROM slots WHERE status = 'accepted' AND id IN ({placeholders})",
                chunk,
            )
            accepted.update(str(row["id"]) for row in rows)
        return accepted

    def generated_candidates(self) -> Iterator[SFTCandidate]:
        rows = self.connection.execute(
            "SELECT candidate_json FROM attempts WHERE status = 'generated' AND candidate_json IS NOT NULL "
            "ORDER BY slot_id, attempt_no"
        )
        for row in rows:
            yield SFTCandidate.model_validate_json(row["candidate_json"])

    def generated_candidates_batch(self, limit: int) -> Iterator[SFTCandidate]:
        rows = self.connection.execute(
            "SELECT a.candidate_json FROM attempts a JOIN slots s ON s.id = a.slot_id "
            "WHERE a.status = 'generated' AND a.candidate_json IS NOT NULL "
            "ORDER BY s.ordinal, a.attempt_no LIMIT ?",
            (limit,),
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
                self.connection.execute(
                    "INSERT OR IGNORE INTO accepted_fingerprints(fingerprint, candidate_id) VALUES (?, ?)",
                    (candidate_fingerprint(candidate.instruction, candidate.input, candidate.output), candidate.id),
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
            "SELECT COUNT(*) total, SUM(status = 'accepted') accepted, SUM(status != 'accepted') pending FROM slots"
        ).fetchone()
        attempts = self.connection.execute(
            "SELECT COUNT(*) total, SUM(status = 'reject') rejected, SUM(status = 'review') reviewed, "
            "SUM(status = 'error') errors, SUM(status = 'generated') generated, "
            "SUM(speculative = 1) speculative FROM attempts"
        ).fetchone()
        llm_judged = self.connection.execute(
            "SELECT COUNT(*) FROM attempts WHERE evaluation_json LIKE '%\"selected_for_llm\":true%'"
        ).fetchone()[0]
        return {
            "target": int(slots["total"] or 0),
            "accepted": int(slots["accepted"] or 0),
            "pending": int(slots["pending"] or 0),
            "attempted": int(attempts["total"] or 0),
            "rejected": int(attempts["rejected"] or 0),
            "reviewed": int(attempts["reviewed"] or 0),
            "errors": int(attempts["errors"] or 0),
            "generated": int(attempts["generated"] or 0),
            "speculative": int(attempts["speculative"] or 0),
            "llm_judged": int(llm_judged or 0),
        }

    def progress_counts(self, max_attempts: int) -> dict[str, int]:
        counts = self.counts()
        exhausted = self.connection.execute(
            "SELECT COUNT(*) FROM slots WHERE status != 'accepted' AND attempt_count >= ?",
            (max_attempts,),
        ).fetchone()[0]
        counts["exhausted"] = int(exhausted or 0)
        return counts

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

    def set_metadata(self, key: str, value: object) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO run_metadata(key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP",
                (key, json.dumps(value, ensure_ascii=False)),
            )

    def get_metadata(self, key: str, default: object | None = None) -> object | None:
        row = self.connection.execute("SELECT value FROM run_metadata WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def record_document_shard(self, path: str, document_count: int, byte_count: int) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO document_shards(path, document_count, byte_count) VALUES (?, ?, ?)",
                (path, document_count, byte_count),
            )

    def record_document(
        self,
        *,
        document_id: str,
        source: str,
        title: str | None,
        estimated_tokens: int,
        stratum: list[str],
        metadata: dict,
        role: str,
        shard_path: str,
        byte_offset: int,
        byte_length: int,
        ordinal: int,
    ) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO documents("
                "id, source, title, estimated_tokens, stratum_json, metadata_json, role, "
                "shard_path, byte_offset, byte_length, ordinal"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    document_id,
                    source,
                    title,
                    estimated_tokens,
                    json.dumps(stratum, ensure_ascii=False),
                    json.dumps(metadata, ensure_ascii=False),
                    role,
                    shard_path,
                    byte_offset,
                    byte_length,
                    ordinal,
                ),
            )

    def record_documents(self, rows: list[dict]) -> None:
        if not rows:
            return
        with self.connection:
            self.connection.executemany(
                "INSERT OR REPLACE INTO documents("
                "id, source, title, estimated_tokens, stratum_json, metadata_json, role, "
                "shard_path, byte_offset, byte_length, ordinal"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        row["document_id"],
                        row["source"],
                        row["title"],
                        row["estimated_tokens"],
                        json.dumps(row["stratum"], ensure_ascii=False),
                        json.dumps(row["metadata"], ensure_ascii=False),
                        row["role"],
                        row["shard_path"],
                        row["byte_offset"],
                        row["byte_length"],
                        row["ordinal"],
                    )
                    for row in rows
                ],
            )

    def document_location(self, document_id: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()

    def document_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] or 0)

    def reserve_document_ids(self) -> list[str]:
        return [
            str(row["id"])
            for row in self.connection.execute("SELECT id FROM documents WHERE role = 'reserve' ORDER BY ordinal")
        ]

    def slot_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM slots").fetchone()[0] or 0)

    def accepted_fingerprint_exists(self, fingerprint: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM accepted_fingerprints WHERE fingerprint = ?",
            (fingerprint,),
        ).fetchone()
        return row is not None

    def next_checkpoint_index(self, format_name: str, split: str) -> int:
        row = self.connection.execute(
            "SELECT MAX(shard_index) value FROM checkpoint_shards WHERE format = ? AND split = ?",
            (format_name, split),
        ).fetchone()
        value = row["value"]
        return (int(value) if value is not None else -1) + 1

    def checkpoint_item_exists(self, candidate_id: str, format_name: str, split: str = "train") -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM checkpoint_items WHERE candidate_id = ? AND format = ? AND split = ?",
            (candidate_id, format_name, split),
        ).fetchone()
        return row is not None

    def record_checkpoint_shard(
        self,
        *,
        format_name: str,
        split: str,
        shard_index: int,
        path: str,
        row_count: int,
        items: list[tuple[str, int]],
    ) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO checkpoint_shards(format, split, shard_index, path, row_count, status) "
                "VALUES (?, ?, ?, ?, ?, 'closed')",
                (format_name, split, shard_index, path, row_count),
            )
            self.connection.executemany(
                "INSERT OR IGNORE INTO checkpoint_items(candidate_id, format, split, shard_path, row_index) "
                "VALUES (?, ?, ?, ?, ?)",
                [(candidate_id, format_name, split, path, row_index) for candidate_id, row_index in items],
            )

    def checkpoint_shards(self) -> list[dict]:
        rows = self.connection.execute(
            "SELECT format, split, shard_index, path, row_count, status, closed_at "
            "FROM checkpoint_shards ORDER BY format, split, shard_index"
        ).fetchall()
        return [dict(row) for row in rows]

    def recent_attempts(self, limit: int = 20) -> list[dict]:
        rows = self.connection.execute(
            "SELECT candidate_id, slot_id, attempt_no, document_id, status, error, input_tokens, output_tokens, "
            "latency_seconds, created_at FROM attempts ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def recent_errors(self, limit: int = 20) -> list[dict]:
        rows = self.connection.execute(
            "SELECT candidate_id, slot_id, attempt_no, document_id, status, error, created_at "
            "FROM attempts WHERE status = 'error' OR error IS NOT NULL ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def recent_accepted(self, limit: int = 20) -> list[SFTCandidate]:
        rows = self.connection.execute(
            "SELECT accepted_candidate FROM slots WHERE status = 'accepted' "
            "ORDER BY ordinal DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [SFTCandidate.model_validate_json(row["accepted_candidate"]) for row in rows]
