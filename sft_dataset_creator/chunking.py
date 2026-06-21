from __future__ import annotations

from sft_dataset_creator.models import Document, Section


def split_text(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return [""]
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        hard_end = min(len(text), start + size)
        end = hard_end
        if hard_end < len(text):
            minimum = start + size // 2
            paragraph_end = text.rfind("\n\n", minimum, hard_end)
            sentence_end = text.rfind(". ", minimum, hard_end)
            boundary = max(paragraph_end + 2 if paragraph_end >= 0 else -1, sentence_end + 1 if sentence_end >= 0 else -1)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        next_start = max(start + 1, end - overlap)
        while next_start < end and not text[next_start].isspace():
            next_start += 1
        start = next_start
        while start < len(text) and text[start].isspace():
            start += 1
    return chunks


def chunk_document(document: Document, size: int, overlap: int) -> Document:
    source_sections = document.sections or [Section(id="0", title=document.title, text=document.text)]
    chunks: list[Section] = []
    for section in source_sections:
        for text in split_text(section.text, size, overlap):
            if text:
                chunks.append(Section(id=str(len(chunks)), title=section.title, text=text))
    if not chunks:
        chunks.append(Section(id="0", title=document.title, text=""))
    return document.model_copy(
        update={
            "sections": chunks,
            "metadata": {
                **document.metadata,
                "chunking": {"size_characters": size, "overlap_characters": overlap},
            },
        }
    )


def select_chunk(document: Document, chunk_id: str | None, fallback_index: int = 0) -> Document:
    if chunk_id is None:
        return document
    sections = document.evidence_sections()
    selected = next((section for section in sections if section.id == chunk_id), None)
    if selected is None:
        selected = sections[fallback_index % len(sections)]
    return document.model_copy(
        update={
            "sections": [selected],
            "metadata": {**document.metadata, "selected_chunk_id": selected.id},
        }
    )
