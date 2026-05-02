from __future__ import annotations

import re
import unicodedata
from typing import Any

from wiki_if_builder.schemas import TriageResult


TITLE_PREFIX_RE = re.compile(
    r"^\s*(categoria|predefini[cç][aã]o|ficheiro|wikip[eé]dia|ajuda|portal)\s*:",
    flags=re.IGNORECASE,
)
REDIRECT_RE = re.compile(r"^\s*#\s*(redirect|redirecionamento)", flags=re.IGNORECASE)
DISAMBIG_RE = re.compile(r"(desambigua[cç][aã]o|disambiguation)", flags=re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+", flags=re.UNICODE)
ALPHA_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]", flags=re.UNICODE)


VALID_STATUSES = {"valid_article", "valid_short_article"}


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _words(text: str) -> list[str]:
    return [word.lower() for word in WORD_RE.findall(text)]


def _symbolic_ratio(text: str) -> float:
    if not text:
        return 1.0
    symbolic = 0
    for char in text:
        category = unicodedata.category(char)
        if category.startswith("P") or category.startswith("S"):
            symbolic += 1
    return symbolic / len(text)


def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(ALPHA_RE.findall(text)) / len(text)


def _is_list_like(title: str, text: str) -> bool:
    if title.strip().lower().startswith(("lista de ", "lista dos ", "lista das ")):
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 8:
        return False
    bullet_lines = sum(
        1
        for line in lines
        if line.startswith(("*", "-", "•")) or re.match(r"^\d+[\).]\s+", line) is not None
    )
    return bullet_lines / len(lines) >= 0.5


def _is_stub_like(text: str) -> bool:
    lowered = text.lower()
    return len(text) < 900 or "este artigo é um esboço" in lowered or "artigo mínimo" in lowered


def _quality_metrics(text: str) -> tuple[int, int, int, float, float]:
    words = _words(text)
    return (
        len(text),
        len(words),
        len(set(words)),
        _alpha_ratio(text),
        _symbolic_ratio(text),
    )


def triage_article(article: dict[str, Any]) -> TriageResult:
    title = _as_text(article.get("title")).strip()
    text = _as_text(article.get("text")).strip()
    ns = article.get("ns", 0)

    char_count, word_count, unique_word_count, alpha_ratio, symbolic_ratio = _quality_metrics(text)
    base = {
        "char_count": char_count,
        "word_count": word_count,
        "unique_word_count": unique_word_count,
        "alpha_ratio": alpha_ratio,
        "symbolic_ratio": symbolic_ratio,
        "used_char_count": char_count,
        "original_char_count": char_count,
    }

    try:
        namespace = int(ns)
    except (TypeError, ValueError):
        namespace = -1

    if namespace != 0:
        return TriageResult(
            status="non_main_namespace",
            reason=f"Namespace não principal: {ns}",
            quality_flags=["non_main_namespace"],
            **base,
        )
    if TITLE_PREFIX_RE.search(title):
        return TriageResult(
            status="non_main_namespace",
            reason="Título pertence a namespace não principal",
            quality_flags=["non_main_namespace", "title_namespace_prefix"],
            **base,
        )
    if not text or char_count < 300:
        return TriageResult(
            status="empty_or_symbolic",
            reason="Texto vazio ou curto demais para triagem confiável",
            quality_flags=["empty_or_too_short"],
            **base,
        )
    if REDIRECT_RE.search(text[:300]):
        return TriageResult(
            status="redirect_like",
            reason="Texto parece redirect",
            quality_flags=["redirect_like"],
            **base,
        )
    if DISAMBIG_RE.search(title) or DISAMBIG_RE.search(text[:1500]):
        return TriageResult(
            status="disambiguation_like",
            reason="Texto parece página de desambiguação",
            quality_flags=["disambiguation_like"],
            **base,
        )
    if alpha_ratio < 0.35 or symbolic_ratio > 0.45:
        return TriageResult(
            status="empty_or_symbolic",
            reason="Texto com baixa proporção de letras ou alta proporção de símbolos",
            quality_flags=["symbolic_garbage"],
            **base,
        )
    if _is_list_like(title, text):
        return TriageResult(
            status="list_like",
            reason="Texto parece lista, não artigo enciclopédico narrativo",
            quality_flags=["list_like"],
            **base,
        )
    if word_count < 60 or unique_word_count < 35 or (word_count and unique_word_count / word_count < 0.08):
        return TriageResult(
            status="low_signal",
            reason="Texto com baixa diversidade lexical",
            quality_flags=["low_signal"],
            **base,
        )

    flags: list[str] = []
    status = "valid_article"
    reason = "Artigo válido"
    if _is_stub_like(text):
        status = "valid_short_article"
        flags.append("stub_like")
        reason = "Artigo curto, aceito com cautela"

    return TriageResult(status=status, reason=reason, quality_flags=flags, **base)


def is_valid_for_llm(triage: TriageResult) -> bool:
    return triage.status in VALID_STATUSES
