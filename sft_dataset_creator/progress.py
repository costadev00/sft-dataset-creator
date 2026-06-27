from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from sft_dataset_creator.config import ProjectConfig
from sft_dataset_creator.models import GenerationRequest
from sft_dataset_creator.state import RunState


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


class ProgressReporter:
    def __init__(self, run_dir: str | Path, config: ProjectConfig, state: RunState) -> None:
        self.run_dir = Path(run_dir)
        self.config = config
        self.state = state
        self.path = self.run_dir / "progress.json"
        self.started = time.time()
        self.last_write = 0.0
        self.phase = "starting"
        self.status = "running"
        self.current: dict[str, Any] = {}
        self.gpu: dict[str, Any] = {}
        self.recent_rates: list[tuple[float, int, int]] = []

    def set_phase(self, phase: str, status: str | None = None, current: dict[str, Any] | None = None) -> None:
        self.phase = phase
        if status is not None:
            self.status = status
        if current is not None:
            self.current = current
        elif phase == "finished":
            self.current = {}
        self.write(force=True)

    def set_current_request(self, request: GenerationRequest) -> None:
        self.current = {
            "current_slot_id": request.slot_id,
            "current_document_id": request.document_id,
            "current_title": request.metadata.get("document_title"),
            "task": request.task,
            "difficulty": request.difficulty,
            "attempt": str(request.request_id or "").rsplit("-a", 1)[-1],
        }
        self.write()

    def set_gpu_stats(self, gpu: dict[str, Any]) -> None:
        self.gpu = gpu
        self.write(force=True)

    def write(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_write < self.config.runtime.checkpoint.progress_interval_seconds:
            return
        counts = self.state.progress_counts(self.config.target.max_attempts_per_slot)
        accepted = counts["accepted"]
        attempted = counts["attempted"]
        target = counts["target"] or self.config.target.examples
        elapsed = max(0.001, now - self.started)
        self.recent_rates.append((now, accepted, attempted))
        self.recent_rates = [item for item in self.recent_rates if now - item[0] <= 900]
        first = self.recent_rates[0]
        window = max(0.001, now - first[0])
        accepted_rate = (accepted - first[1]) / window if len(self.recent_rates) > 1 else accepted / elapsed
        attempt_rate = (attempted - first[2]) / window if len(self.recent_rates) > 1 else attempted / elapsed
        remaining = max(0, target - accepted)
        terminal = self.phase == "finished" or self.status in {"completed", "partial", "interrupted", "failed"}
        eta_seconds = None if terminal else remaining / accepted_rate if accepted_rate > 0 else None
        payload = {
            "phase": self.phase,
            "status": self.status,
            **self.current,
            "target": target,
            "accepted": accepted,
            "attempted": attempted,
            "rejected": counts["rejected"],
            "reviewed": counts["reviewed"],
            "errors": counts["errors"],
            "pending": counts["pending"],
            "exhausted": counts["exhausted"],
            "generated_waiting_evaluation": counts["generated"],
            "accepted_percent": round(accepted * 100 / target, 3) if target else 0.0,
            "attempts_percent": round(
                attempted * 100 / max(1, int(target * self.config.target.max_total_attempt_multiplier)),
                3,
            ),
            "accepted_per_minute": round(accepted_rate * 60, 3),
            "attempts_per_minute": round(attempt_rate * 60, 3),
            "eta_seconds": round(eta_seconds, 3) if eta_seconds is not None else None,
            "elapsed_seconds": round(elapsed, 3),
            "checkpoint_shards": self.state.checkpoint_shards(),
            "recent_errors": self.state.recent_errors(10),
            "gpu": self.gpu,
            "updated_at": now,
        }
        _atomic_json(self.path, payload)
        self.last_write = now
