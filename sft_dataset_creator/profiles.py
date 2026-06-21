from __future__ import annotations

import re

from sft_dataset_creator.models import Document


WIKI_TITLE_PREFIX = re.compile(
    r"^\s*(category|categoria|template|predefini[cç][aã]o|file|ficheiro|wikip[eé]dia|help|ajuda|portal)\s*:",
    re.IGNORECASE,
)


def document_is_eligible(document: Document, profile: str | None) -> bool:
    if not document.text.strip():
        return False
    if profile != "wikipedia_ptbr":
        return True
    namespace = document.metadata.get("ns", document.metadata.get("namespace", 0))
    try:
        if int(namespace) != 0:
            return False
    except (TypeError, ValueError):
        return False
    title = document.title or ""
    start = document.text[:1500].lower()
    if WIKI_TITLE_PREFIX.search(title):
        return False
    if re.match(r"^\s*#\s*(redirect|redirecionamento)", start, re.IGNORECASE):
        return False
    if "desambiguacao" in start or "desambiguação" in start or "desambiguação" in title.lower():
        return False
    return len(document.text) >= 300
