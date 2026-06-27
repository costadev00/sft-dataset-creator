from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from sft_dataset_creator.models import Document
from sft_dataset_creator.planner import load_document_snapshot
from sft_dataset_creator.state import RunState


class DocumentStore:
    def get(self, document_id: str) -> Document:
        raise NotImplementedError


class LegacyDocumentStore(DocumentStore):
    def __init__(self, snapshot_path: str | Path) -> None:
        self.documents = load_document_snapshot(snapshot_path)

    def get(self, document_id: str) -> Document:
        return self.documents[document_id]


class SQLiteDocumentStore(DocumentStore):
    def __init__(self, run_dir: str | Path, state: RunState, cache_size: int = 2_048) -> None:
        self.run_dir = Path(run_dir)
        self.state = state
        self.cache_size = cache_size
        self.cache: OrderedDict[str, Document] = OrderedDict()

    def get(self, document_id: str) -> Document:
        cached = self.cache.get(document_id)
        if cached is not None:
            self.cache.move_to_end(document_id)
            return cached
        location = self.state.document_location(document_id)
        if location is None:
            raise KeyError(f"unknown document id: {document_id}")
        path = Path(str(location["shard_path"]))
        if not path.is_absolute():
            path = self.run_dir / path
        with path.open("rb") as handle:
            handle.seek(int(location["byte_offset"]))
            payload = handle.read(int(location["byte_length"]))
        document = Document.model_validate_json(payload.decode("utf-8").strip())
        self.cache[document_id] = document
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return document
