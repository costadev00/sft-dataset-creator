from __future__ import annotations

from sft_dataset_creator.registry import create
from sft_dataset_creator.config import SourceConfig


def test_local_source_maps_fields(project_config) -> None:
    source = create("sources", project_config.source.plugin, project_config.source)
    documents = list(source.iter_documents())
    assert len(documents) == 8
    assert documents[0].id == "doc-0"
    assert documents[0].metadata["domain"] == "science"


def test_registry_lists_builtin_plugins() -> None:
    from sft_dataset_creator.registry import available_plugins

    plugins = available_plugins()
    assert "local" in plugins["sources"]
    assert "vllm_local" in plugins["backends"]
    assert "messages" in plugins["exporters"]


def test_local_parquet_source(tmp_path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "corpus.parquet"
    pq.write_table(pa.Table.from_pylist([{"id": "p1", "text": "Parquet document text."}]), path)
    config = SourceConfig(plugin="local", params={"path": str(path)}, field_map={"id": "id", "text": "text"})
    documents = list(create("sources", "local", config).iter_documents())
    assert documents[0].id == "p1"


def test_huggingface_source_uses_field_mapping(monkeypatch) -> None:
    import datasets

    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *_args, **_kwargs: [{"page_id": 7, "body": "Mapped Hugging Face text", "heading": "Title"}],
    )
    config = SourceConfig(
        plugin="huggingface",
        params={"dataset": "owner/example", "split": "train"},
        field_map={"id": "page_id", "text": "body", "title": "heading"},
    )
    document = next(create("sources", "huggingface", config).iter_documents())
    assert document.id == "7"
    assert document.title == "Title"
