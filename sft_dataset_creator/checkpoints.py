from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sft_dataset_creator.config import ProjectConfig
from sft_dataset_creator.exporters import AlpacaExporter, MessagesExporter, PromptCompletionExporter
from sft_dataset_creator.models import SFTCandidate
from sft_dataset_creator.state import RunState


class CheckpointWriter:
    def __init__(self, run_dir: str | Path, state: RunState, config: ProjectConfig) -> None:
        self.run_dir = Path(run_dir)
        self.state = state
        self.config = config
        self.root = self.run_dir / "checkpoints"
        self.formats = list(config.runtime.checkpoint.formats)
        self.shard_size = config.runtime.checkpoint.shard_size
        self.split = "train"
        self._active: dict[str, dict[str, Any]] = {}
        self._exporters = {
            "messages": MessagesExporter(),
            "prompt_completion": PromptCompletionExporter(),
            "alpaca": AlpacaExporter(),
        }
        self.cleanup_incomplete()

    def cleanup_incomplete(self) -> None:
        if not self.root.exists():
            return
        for path in self.root.glob("*/*/*.tmp"):
            path.unlink(missing_ok=True)

    def _row(self, format_name: str, candidate: SFTCandidate) -> dict[str, Any]:
        if format_name == "canonical":
            return candidate.model_dump(mode="json")
        return self._exporters[format_name].render(candidate)

    def _open(self, format_name: str) -> dict[str, Any]:
        active = self._active.get(format_name)
        if active is not None:
            return active
        shard_index = self.state.next_checkpoint_index(format_name, self.split)
        directory = self.root / format_name / self.split
        directory.mkdir(parents=True, exist_ok=True)
        final_path = directory / f"part-{shard_index:06d}.jsonl"
        tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
        handle = tmp_path.open("w", encoding="utf-8")
        active = {
            "index": shard_index,
            "final_path": final_path,
            "tmp_path": tmp_path,
            "handle": handle,
            "count": 0,
            "items": [],
        }
        self._active[format_name] = active
        return active

    def write(self, candidate: SFTCandidate) -> None:
        for format_name in self.formats:
            if self.state.checkpoint_item_exists(candidate.id, format_name, self.split):
                continue
            active = self._open(format_name)
            row_index = int(active["count"])
            active["handle"].write(json.dumps(self._row(format_name, candidate), ensure_ascii=False) + "\n")
            active["count"] = row_index + 1
            active["items"].append((candidate.id, row_index))
            if int(active["count"]) >= self.shard_size:
                self._finalize(format_name)

    def _finalize(self, format_name: str) -> None:
        active = self._active.pop(format_name, None)
        if active is None:
            return
        handle = active["handle"]
        handle.flush()
        handle.close()
        if int(active["count"]) == 0:
            Path(active["tmp_path"]).unlink(missing_ok=True)
            return
        Path(active["tmp_path"]).replace(active["final_path"])
        self.state.record_checkpoint_shard(
            format_name=format_name,
            split=self.split,
            shard_index=int(active["index"]),
            path=str(Path(active["final_path"]).relative_to(self.run_dir)),
            row_count=int(active["count"]),
            items=list(active["items"]),
        )

    def close(self) -> None:
        for format_name in list(self._active):
            self._finalize(format_name)

    def catch_up(self) -> int:
        written = 0
        for candidate in self.state.accepted_candidates():
            missing = [
                format_name
                for format_name in self.formats
                if not self.state.checkpoint_item_exists(candidate.id, format_name, self.split)
            ]
            if not missing:
                continue
            self.write(candidate)
            written += 1
        self.close()
        return written
