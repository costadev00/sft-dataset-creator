from __future__ import annotations

import re
import unicodedata


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(character for character in value if not unicodedata.combining(character))
    return " ".join(value.casefold().split())


_SOURCE = r"(?:texto|documento|contexto|trecho|passagem|artigo|conteudo|material|fonte)"
_PORTUGUESE_PATTERNS = (
    rf"\b(?:de acordo com|com base (?:em|no|na)|segundo|conforme)\s+"
    rf"(?:(?:o|a|este|esta|esse|essa|um|uma)\s+)?{_SOURCE}\b",
    rf"\b{_SOURCE}\s+(?:acima|abaixo|fornecid[oa]s?|apresentad[oa]s?)\b",
    rf"\b(?:citad[oa]s?|mencionad[oa]s?|descrit[oa]s?|apresentad[oa]s?|informad[oa]s?)\s+"
    rf"(?:neste|nesta|nesse|nessa|no|na)\s+{_SOURCE}\b",
    rf"\b(?:o|a|este|esta|esse|essa)\s+{_SOURCE}\s+"
    r"(?:afirma|apresenta|cita|descreve|fornece|informa|menciona|mostra)\b",
)
_ENGLISH_PATTERNS = (
    r"\b(?:according to|based on)\s+(?:(?:the|this|that|provided|above|below)\s+)?"
    r"(?:text|document|context|passage|article|content|material|source)\b",
    r"\b(?:text|document|context|passage|article|content|material|source)\s+"
    r"(?:above|below|provided)\b",
    r"\b(?:cited|mentioned|described|presented|provided|stated|shown)\s+in\s+"
    r"(?:(?:the|this|that|provided)\s+)?(?:text|document|context|passage|article|source)\b",
)
_SPANISH_PATTERNS = (
    r"\b(?:segun|de acuerdo con|con base en)\s+(?:(?:el|la|este|esta|ese|esa)\s+)?"
    r"(?:texto|documento|contexto|pasaje|articulo|contenido|material|fuente)\b",
    r"\b(?:texto|documento|contexto|pasaje|articulo|contenido|material|fuente)\s+"
    r"(?:anterior|siguiente|proporcionad[oa])\b",
)
_SOURCE_REFERENCE_PATTERNS = tuple(
    re.compile(pattern) for pattern in (*_PORTUGUESE_PATTERNS, *_ENGLISH_PATTERNS, *_SPANISH_PATTERNS)
)


def source_reference(value: str) -> str | None:
    """Return the first reference to hidden source material in user-facing text."""
    normalized = _normalize(value)
    for pattern in _SOURCE_REFERENCE_PATTERNS:
        match = pattern.search(normalized)
        if match:
            return match.group(0)
    return None


def candidate_source_reference(*values: str) -> str | None:
    for value in values:
        match = source_reference(value)
        if match:
            return match
    return None
