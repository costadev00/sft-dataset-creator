from __future__ import annotations

from sft_dataset_creator.chunking import chunk_document, select_chunk, split_text
from sft_dataset_creator.models import Document


def test_split_text_creates_overlapping_bounded_chunks() -> None:
    text = "A" * 600 + " " + "B" * 600 + " " + "C" * 600

    chunks = split_text(text, size=800, overlap=100)

    assert len(chunks) == 3
    assert all(len(chunk) <= 800 for chunk in chunks)


def test_chunk_document_assigns_stable_numeric_section_ids() -> None:
    document = Document(id="doc", source="test", text="A" * 1_300)

    chunked = chunk_document(document, size=500, overlap=0)
    selected = select_chunk(chunked, "1")

    assert [section.id for section in chunked.sections] == ["0", "1", "2"]
    assert [section.id for section in selected.sections] == ["1"]
