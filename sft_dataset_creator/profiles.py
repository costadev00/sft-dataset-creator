from __future__ import annotations

import re

from sft_dataset_creator.models import Document


WIKI_TITLE_PREFIX = re.compile(
    r"^\s*(category|categoria|template|predefini[cç][aã]o|file|ficheiro|wikip[eé]dia|"
    r"wikibooks|wikilivros|help|ajuda|portal)\s*:",
    re.IGNORECASE,
)
REDIRECT_PREFIX = re.compile(r"^\s*#\s*(redirect|redirecionamento)", re.IGNORECASE)
PARAGRAPH_SEPARATOR = re.compile(r"\n\s*\n+")
MIN_PARAGRAPH_CHARACTERS = 300


def _has_minimum_paragraph(text: str) -> bool:
    return any(len(paragraph.strip()) >= MIN_PARAGRAPH_CHARACTERS for paragraph in PARAGRAPH_SEPARATOR.split(text))


def _is_common_wiki_content(document: Document) -> bool:
    title = document.title or ""
    start = document.text[:1500].lower()
    if WIKI_TITLE_PREFIX.search(title):
        return False
    if REDIRECT_PREFIX.match(start):
        return False
    if "desambiguacao" in start or "desambiguação" in start or "desambiguação" in title.lower():
        return False
    return _has_minimum_paragraph(document.text)


def document_is_eligible(document: Document, profile: str | None) -> bool:
    if not document.text.strip():
        return False
    if profile is None:
        return True
    if profile == "wikibooks_ptbr":
        return _is_common_wiki_content(document)
    if profile != "wikipedia_ptbr":
        return True
    namespace = document.metadata.get("ns", document.metadata.get("namespace", 0))
    try:
        if int(namespace) != 0:
            return False
    except (TypeError, ValueError):
        return False
    return _is_common_wiki_content(document)
