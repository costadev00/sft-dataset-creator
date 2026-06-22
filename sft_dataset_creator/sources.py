from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from sft_dataset_creator.config import SourceConfig
from sft_dataset_creator.models import Document, Section
from sft_dataset_creator.registry import register


DEFAULT_FIELD_MAP = {
    "id": "id",
    "text": "text",
    "title": "title",
    "sections": "sections",
    "license": "license",
}


def get_nested(record: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = record
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _sections(value: Any) -> list[Section]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    output: list[Section] = []
    for index, item in enumerate(value):
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("content") or item.get("section_text") or "")
            section_id = str(item.get("id") or item.get("section_id") or index)
            title = item.get("title") or item.get("heading")
        else:
            text = str(item or "")
            section_id = str(index)
            title = None
        if text.strip():
            output.append(Section(id=section_id, text=text.strip(), title=str(title) if title else None))
    return output


class MappedSource:
    def __init__(self, config: SourceConfig) -> None:
        self.config = config
        self.field_map = {**DEFAULT_FIELD_MAP, **config.field_map}

    def normalize(self, record: dict[str, Any], index: int) -> Document:
        mapped_paths = set(self.field_map.values())
        document_id = get_nested(record, self.field_map["id"], index)
        text = str(get_nested(record, self.field_map["text"], "") or "").strip()
        title = get_nested(record, self.field_map["title"], None)
        license_name = get_nested(record, self.field_map["license"], None)
        raw_sections = get_nested(record, self.field_map["sections"], None)
        metadata = {key: value for key, value in record.items() if key not in mapped_paths}
        metadata["raw_index"] = index
        return Document(
            id=str(document_id),
            text=text,
            title=str(title) if title is not None else None,
            sections=_sections(raw_sections),
            metadata=metadata,
            license=str(license_name) if license_name else None,
            source=self.config.plugin,
        )


class LocalFileSource(MappedSource):
    def __init__(self, config: SourceConfig) -> None:
        super().__init__(config)
        try:
            self.path = Path(config.params["path"]).expanduser()
        except KeyError as exc:
            raise ValueError("local source requires params.path") from exc
        self.format = str(config.params.get("format") or self.path.suffix.lstrip(".")).lower()

    def _records(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if self.format in {"jsonl", "ndjson"}:
            with self.path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid JSONL at {self.path}:{line_number}: {exc}") from exc
                    if not isinstance(value, dict):
                        raise ValueError(f"JSONL record at {self.path}:{line_number} is not an object")
                    yield value
            return
        if self.format == "json":
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            records = payload if isinstance(payload, list) else payload.get("records", [])
            for record in records:
                if not isinstance(record, dict):
                    raise ValueError("JSON source records must be objects")
                yield record
            return
        if self.format == "parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise ImportError("Parquet sources require the 'hf' extra") from exc
            parquet = pq.ParquetFile(self.path)
            for batch in parquet.iter_batches():
                yield from batch.to_pylist()
            return
        raise ValueError(f"unsupported local source format: {self.format}")

    def iter_documents(self) -> Iterator[Document]:
        for index, record in enumerate(self._records()):
            yield self.normalize(record, index)


class HuggingFaceSource(MappedSource):
    def __init__(self, config: SourceConfig) -> None:
        super().__init__(config)
        self.dataset = str(config.params.get("dataset") or "")
        if not self.dataset:
            raise ValueError("huggingface source requires params.dataset")
        self.split = str(config.params.get("split") or "train")
        self.subset = config.params.get("subset")
        self.streaming = bool(config.params.get("streaming", True))
        self.cache_dir = config.params.get("cache_dir")
        self.revision = config.params.get("revision")

    def iter_documents(self) -> Iterator[Document]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("Hugging Face sources require the 'hf' extra") from exc
        dataset = load_dataset(
            self.dataset,
            self.subset,
            split=self.split,
            streaming=self.streaming,
            cache_dir=self.cache_dir,
            revision=self.revision,
        )
        for index, record in enumerate(dataset):
            yield self.normalize(dict(record), index)


register("sources", "local", lambda config: LocalFileSource(SourceConfig.model_validate(config)))
register("sources", "huggingface", lambda config: HuggingFaceSource(SourceConfig.model_validate(config)))
