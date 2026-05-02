from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from wiki_if_builder.schemas import AnalystOutput, DocumentLabels, EvidenceRef, IFCandidate


ALLOWED_TASK_TYPES = {
    "definition",
    "closed_qa",
    "summarization",
    "information_extraction",
    "comparison",
    "classification",
    "timeline",
    "rewrite",
    "concept_explanation",
    "structured_extraction",
    "fact_checking",
    "taxonomy",
    "short_answer",
    "didactic_explanation",
}
ALLOWED_RISK_BANDS = {"low", "medium", "high", "unknown"}
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard", "unknown"}
RISK_ALIASES = {"baixo": "low", "baixa": "low", "medio": "medium", "media": "medium", "alto": "high", "alta": "high"}
DIFFICULTY_ALIASES = {
    "facil": "easy",
    "easy": "easy",
    "medio": "medium",
    "media": "medium",
    "medium": "medium",
    "dificil": "hard",
    "hard": "hard",
}


def slugify_pt(value: str | None, fallback: str = "unknown") -> str:
    if not value:
        return fallback
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.lower().strip()
    ascii_text = re.sub(r"[^a-z0-9]+", "_", ascii_text)
    ascii_text = re.sub(r"_+", "_", ascii_text).strip("_")
    return ascii_text or fallback


def normalize_identifier(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower().strip()
    ascii_text = re.sub(r"[^a-z0-9_.-]+", "-", ascii_text)
    ascii_text = re.sub(r"-+", "-", ascii_text).strip("-")
    return ascii_text or fallback


def _compact_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _instruction_key(value: str) -> str:
    return slugify_pt(_compact_text(value), fallback="")


def _limit_completion(task_type: str, completion: str) -> str:
    limits = {
        "summarization": 1800,
        "didactic_explanation": 1800,
        "concept_explanation": 1600,
        "structured_extraction": 1400,
        "timeline": 1400,
    }
    max_chars = limits.get(task_type, 1200)
    completion = completion.strip()
    return completion if len(completion) <= max_chars else completion[:max_chars].rstrip()


def _normalize_refs(refs: Iterable[EvidenceRef], fallback: EvidenceRef) -> list[EvidenceRef]:
    output: list[EvidenceRef] = []
    for ref in refs:
        parsed = ref if isinstance(ref, EvidenceRef) else EvidenceRef.model_validate(ref)
        if parsed.char_end <= parsed.char_start:
            parsed = parsed.model_copy(update={"char_end": parsed.char_start + 500})
        output.append(parsed)
    return output or [fallback]


def normalize_document_labels(labels: DocumentLabels, candidate_task_types: list[str]) -> DocumentLabels:
    task_affordances = [slugify_pt(item) for item in labels.task_affordances]
    task_affordances.extend(candidate_task_types)
    deduped_affordances = list(dict.fromkeys(item for item in task_affordances if item))
    risk_band = slugify_pt(labels.risk_band, fallback="medium")
    risk_band = RISK_ALIASES.get(risk_band, risk_band)
    difficulty_band = slugify_pt(labels.difficulty_band, fallback="medium")
    difficulty_band = DIFFICULTY_ALIASES.get(difficulty_band, difficulty_band)
    return DocumentLabels(
        primary_category=slugify_pt(labels.primary_category, fallback="outros"),
        subcategory=slugify_pt(labels.subcategory, fallback="unknown"),
        secondary_categories=list(
            dict.fromkeys(slugify_pt(item, fallback="outros") for item in labels.secondary_categories)
        ),
        document_type=slugify_pt(labels.document_type, fallback="unknown"),
        task_affordances=deduped_affordances,
        if_eligibility=bool(labels.if_eligibility),
        difficulty_band=difficulty_band if difficulty_band in ALLOWED_DIFFICULTIES else "medium",
        grounding_profile=slugify_pt(labels.grounding_profile, fallback="unknown"),
        risk_band=risk_band if risk_band in ALLOWED_RISK_BANDS else "medium",
    )


def normalize_analyst_output(output: AnalystOutput) -> AnalystOutput:
    fallback_ref = output.evidence_refs[0] if output.evidence_refs else EvidenceRef(section_id=0, char_start=0, char_end=500)
    normalized_candidates: list[IFCandidate] = []
    seen_instructions: set[str] = set()
    seen_ids: set[str] = set()

    for idx, candidate in enumerate(output.if_candidates, start=1):
        instruction = _compact_text(candidate.instruction)
        completion = _compact_text(candidate.completion)
        input_text = _compact_text(candidate.input)
        if not instruction or not completion:
            continue
        key = _instruction_key(instruction)
        if not key or key in seen_instructions:
            continue
        seen_instructions.add(key)

        task_type = slugify_pt(candidate.task_type, fallback="closed_qa")
        if task_type not in ALLOWED_TASK_TYPES:
            task_type = "closed_qa"
        difficulty = slugify_pt(candidate.difficulty, fallback="medium")
        difficulty = DIFFICULTY_ALIASES.get(difficulty, difficulty)
        if difficulty not in ALLOWED_DIFFICULTIES:
            difficulty = "medium"
        style = slugify_pt(candidate.style, fallback="didatico")

        candidate_id = normalize_identifier(candidate.candidate_id, fallback=f"{output.record_id}-c{idx}")
        if candidate_id in seen_ids or not candidate_id.startswith(output.record_id.lower()):
            candidate_id = f"{output.record_id.lower()}-c{idx}"
        suffix = 2
        unique_candidate_id = candidate_id
        while unique_candidate_id in seen_ids:
            unique_candidate_id = f"{candidate_id}-{suffix}"
            suffix += 1
        seen_ids.add(unique_candidate_id)

        normalized_candidates.append(
            IFCandidate(
                candidate_id=unique_candidate_id,
                task_type=task_type,
                instruction=instruction,
                input=input_text,
                completion=_limit_completion(task_type, completion),
                style=style,
                difficulty=difficulty,
                source_refs=_normalize_refs(candidate.source_refs, fallback_ref),
            )
        )

    labels = normalize_document_labels(
        output.document_labels,
        [candidate.task_type for candidate in normalized_candidates],
    )
    normalized_evidence_refs = _normalize_refs(output.evidence_refs, fallback_ref)
    return output.model_copy(
        update={
            "document_labels": labels,
            "evidence_refs": normalized_evidence_refs,
            "if_candidates": normalized_candidates,
        }
    )
