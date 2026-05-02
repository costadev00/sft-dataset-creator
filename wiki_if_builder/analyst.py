from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from wiki_if_builder.config import DEFAULT_LICENSE, AppConfig
from wiki_if_builder.llm_client import LLMJSONParseError, RoundRobinLLMClient
from wiki_if_builder.prompts import DOCUMENT_ANALYST_SYSTEM_PROMPT, render_analyst_user_prompt
from wiki_if_builder.schemas import (
    AnalystOutput,
    DocumentLabels,
    EvidenceRef,
    IFCandidate,
    SourceMetadata,
    TriageResult,
)
from wiki_if_builder.utils import normalize_page_id, record_id_for, truncate_text


@dataclass(slots=True)
class ContextBuildResult:
    context_text: str
    requires_section_fallback: bool
    context_truncated: bool
    used_char_count: int
    original_char_count: int
    evidence_refs: list[EvidenceRef]
    section_texts: list[str]


def _section_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("text") or value.get("section_text") or value.get("content") or "")
    return str(value or "")


def get_article_sections(article: dict[str, Any]) -> list[str]:
    raw_sections = article.get("section_texts") or []
    sections = [_section_text(section).strip() for section in raw_sections]
    sections = [section for section in sections if section]
    if sections:
        return sections
    text = str(article.get("text") or "").strip()
    return [text] if text else []


def build_article_context(article: dict[str, Any], max_input_chars: int) -> ContextBuildResult:
    text = str(article.get("text") or "").strip()
    sections = get_article_sections(article)
    original_char_count = len(text)

    if len(text) <= max_input_chars:
        context_text = f"[section_id=0 char_start=0 char_end={len(text)}]\n{text}"
        return ContextBuildResult(
            context_text=context_text,
            requires_section_fallback=False,
            context_truncated=False,
            used_char_count=len(text),
            original_char_count=original_char_count,
            evidence_refs=[EvidenceRef(section_id=0, char_start=0, char_end=min(len(text), 1000))],
            section_texts=[text],
        )

    if sections and not (len(sections) == 1 and sections[0] == text):
        blocks: list[str] = []
        refs: list[EvidenceRef] = []
        used = 0
        for idx, section in enumerate(sections):
            if used >= max_input_chars:
                break
            remaining = max_input_chars - used
            if remaining <= 0:
                break
            section_used = truncate_text(section, remaining)
            if not section_used:
                continue
            blocks.append(
                f"[section_id={idx} char_start=0 char_end={len(section_used)}]\n{section_used}"
            )
            refs.append(EvidenceRef(section_id=idx, char_start=0, char_end=min(len(section_used), 1000)))
            used += len(section_used)
            if len(section_used) < len(section):
                break
        return ContextBuildResult(
            context_text="\n\n".join(blocks),
            requires_section_fallback=True,
            context_truncated=used < sum(len(section) for section in sections),
            used_char_count=used,
            original_char_count=original_char_count,
            evidence_refs=refs or [EvidenceRef(section_id=0, char_start=0, char_end=0)],
            section_texts=sections,
        )

    truncated = truncate_text(text, max_input_chars)
    return ContextBuildResult(
        context_text=f"[section_id=0 char_start=0 char_end={len(truncated)}]\n{truncated}",
        requires_section_fallback=False,
        context_truncated=True,
        used_char_count=len(truncated),
        original_char_count=original_char_count,
        evidence_refs=[EvidenceRef(section_id=0, char_start=0, char_end=min(len(truncated), 1000))],
        section_texts=[text],
    )


def apply_context_to_triage(triage: TriageResult, context: ContextBuildResult) -> TriageResult:
    return triage.model_copy(
        update={
            "requires_section_fallback": context.requires_section_fallback,
            "context_truncated": context.context_truncated,
            "used_char_count": context.used_char_count,
            "original_char_count": context.original_char_count,
        }
    )


def analyze_article(
    article: dict[str, Any],
    triage_result: TriageResult,
    config: AppConfig,
    llm_client: RoundRobinLLMClient,
) -> AnalystOutput:
    page_id = normalize_page_id(article.get("page_id"))
    record_id = record_id_for(page_id)
    title = str(article.get("title") or "")
    context = build_article_context(article, config.max_input_chars)
    triage_with_context = apply_context_to_triage(triage_result, context)
    source = SourceMetadata(
        dataset=config.dataset_name,
        page_id=page_id,
        title=title,
        license=DEFAULT_LICENSE,
    )

    user_prompt = render_analyst_user_prompt(
        record_id=record_id,
        page_id=page_id,
        title=title,
        dataset_name=config.dataset_name,
        license_name=DEFAULT_LICENSE,
        triage=triage_with_context.model_dump(),
        context_text=context.context_text,
        candidates_per_article=config.candidates_per_article,
    )
    payload = llm_client.chat_json(
        [
            {"role": "system", "content": DOCUMENT_ANALYST_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model=config.model_name,
        temperature=0.1,
        max_tokens=config.max_output_tokens,
    )
    payload["record_id"] = payload.get("record_id") or record_id
    payload["source"] = source.model_dump()
    payload["triage"] = triage_with_context.model_dump()
    payload["model_name"] = config.model_name
    payload["pipeline_version"] = config.pipeline_version
    if not payload.get("evidence_refs"):
        payload["evidence_refs"] = [ref.model_dump() for ref in context.evidence_refs]
    try:
        return AnalystOutput.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"AnalystOutput inválido para page_id={page_id}: {exc}") from exc


def build_synthetic_analyst_output(
    article: dict[str, Any],
    triage_result: TriageResult,
    config: AppConfig,
) -> AnalystOutput:
    page_id = normalize_page_id(article.get("page_id"))
    record_id = record_id_for(page_id)
    title = str(article.get("title") or "")
    context = build_article_context(article, config.max_input_chars)
    triage_with_context = apply_context_to_triage(triage_result, context)
    snippet = truncate_text(str(article.get("text") or ""), 900)
    source = SourceMetadata(
        dataset=config.dataset_name,
        page_id=page_id,
        title=title,
        license=DEFAULT_LICENSE,
    )
    primary = "ciencia"
    subcategory = "conhecimento_geral"
    document_type = "encyclopedic_article"
    if "machado" in title.lower():
        primary = "literatura"
        subcategory = "literatura_brasileira"
        document_type = "biography"
    elif "amaz" in title.lower():
        primary = "geografia"
        subcategory = "hidrografia"
        document_type = "geographical_article"

    labels = DocumentLabels(
        primary_category=primary,
        subcategory=subcategory,
        secondary_categories=["educacao"],
        document_type=document_type,
        task_affordances=["definition", "closed_qa", "summarization"],
        if_eligibility=triage_with_context.status == "valid_article",
        difficulty_band="medium",
        grounding_profile="strong",
        risk_band="low",
    )
    refs = context.evidence_refs or [EvidenceRef(section_id=0, char_start=0, char_end=min(len(snippet), 500))]
    candidates = [
        IFCandidate(
            candidate_id=f"{record_id}-c1",
            task_type="definition",
            instruction=f"Explique, de forma didática, o tema central do artigo sobre {title}.",
            input=truncate_text(snippet, 650),
            completion=f"O artigo apresenta {title} como tema enciclopédico e destaca seus conceitos principais com base no contexto fornecido.",
            style="didatico",
            difficulty="easy",
            source_refs=refs[:1],
        ),
        IFCandidate(
            candidate_id=f"{record_id}-c2",
            task_type="summarization",
            instruction=f"Resuma as informações principais do artigo sobre {title}.",
            input=truncate_text(snippet, 850),
            completion=f"Em síntese, o artigo descreve {title}, contextualiza sua relevância e reúne informações factuais úteis para estudo.",
            style="objetivo",
            difficulty="medium",
            source_refs=refs[:1],
        ),
        IFCandidate(
            candidate_id=f"{record_id}-c3",
            task_type="information_extraction",
            instruction=f"Extraia dois pontos relevantes mencionados no artigo sobre {title}.",
            input=truncate_text(snippet, 850),
            completion=f"1. O artigo identifica {title} como o assunto principal. 2. O texto apresenta informações contextuais e enciclopédicas sobre o tema.",
            style="estruturado",
            difficulty="medium",
            source_refs=refs[:1],
        ),
    ][: max(1, config.candidates_per_article)]

    return AnalystOutput(
        record_id=record_id,
        source=source,
        triage=triage_with_context,
        document_labels=labels,
        evidence_refs=refs,
        if_candidates=candidates,
        model_name=config.model_name,
        pipeline_version=config.pipeline_version,
    )


def error_record(stage: str, article: dict[str, Any], error: BaseException) -> dict[str, Any]:
    page_id = normalize_page_id(article.get("page_id"))
    raw_response = getattr(error, "raw_response", None)
    base_url = getattr(error, "base_url", None)
    return {
        "stage": stage,
        "page_id": page_id,
        "title": str(article.get("title") or ""),
        "error_type": type(error).__name__,
        "error": str(error),
        "raw_response": raw_response if isinstance(error, LLMJSONParseError) else raw_response,
        "base_url": base_url,
    }

