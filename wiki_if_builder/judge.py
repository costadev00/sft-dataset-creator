from __future__ import annotations

from typing import Any

from wiki_if_builder.config import AppConfig
from wiki_if_builder.llm_client import RoundRobinLLMClient
from wiki_if_builder.prompts import JUDGE_SYSTEM_PROMPT, render_judge_user_prompt
from wiki_if_builder.schemas import EvidenceRef, IFCandidate, JudgeDecision, JudgeResult, JudgeScores
from wiki_if_builder.utils import normalize_page_id, truncate_text


def _section_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("text") or value.get("section_text") or value.get("content") or "")
    return str(value or "")


def _article_sections(article: dict[str, Any]) -> list[str]:
    raw_sections = article.get("section_texts") or []
    sections = [_section_text(section).strip() for section in raw_sections]
    sections = [section for section in sections if section]
    if sections:
        return sections
    text = str(article.get("text") or "").strip()
    return [text] if text else []


def resolve_evidence_snippets(
    article: dict[str, Any],
    refs: list[EvidenceRef],
    *,
    max_snippet_chars: int = 1200,
) -> list[str]:
    sections = _article_sections(article)
    snippets: list[str] = []
    for ref in refs:
        if ref.section_id < 0 or ref.section_id >= len(sections):
            continue
        section = sections[ref.section_id]
        start = max(0, min(ref.char_start, len(section)))
        end = max(start, min(ref.char_end or start + max_snippet_chars, len(section)))
        if end <= start:
            end = min(len(section), start + max_snippet_chars)
        snippet = section[start:end].strip()
        if snippet:
            snippets.append(
                f"section_id={ref.section_id} char_start={start} char_end={end}\n"
                f"{truncate_text(snippet, max_snippet_chars)}"
            )
    return snippets


def heuristic_judge_candidate(
    candidate: IFCandidate,
    article: dict[str, Any],
    config: AppConfig,
) -> JudgeResult:
    snippets = resolve_evidence_snippets(article, candidate.source_refs)
    issues: list[str] = []
    verdict = "accept"
    needs_review = False
    if not candidate.instruction.strip() or not candidate.completion.strip():
        verdict = "reject"
        issues.append("instruction_or_completion_empty")
    elif not candidate.source_refs or not snippets:
        verdict = "review"
        needs_review = True
        issues.append("insufficient_grounding_evidence")
    elif len(candidate.instruction.split()) < 4:
        verdict = "review"
        needs_review = True
        issues.append("instruction_too_short")
    elif len(candidate.completion) < 25:
        verdict = "review"
        needs_review = True
        issues.append("completion_too_short")

    score = 4.2 if verdict == "accept" else (2.0 if verdict == "reject" else 3.4)
    scores = JudgeScores(
        grounding=4.0 if snippets else 2.0,
        instruction_quality=4.0 if "instruction_too_short" not in issues else 2.5,
        completion_quality=4.0 if "completion_too_short" not in issues else 2.5,
        non_triviality=3.8,
        style_clarity=4.0,
        schema_validity=5.0,
    )
    return JudgeResult(
        candidate_id=candidate.candidate_id,
        source_page_id=normalize_page_id(article.get("page_id")),
        source_title=str(article.get("title") or ""),
        judge_model_name=config.judge_model_name,
        judge=JudgeDecision(
            verdict=verdict,
            overall_score=score,
            scores=scores,
            issues=issues,
            needs_human_review=needs_review or verdict == "review",
        ),
    )


def judge_candidate(
    candidate: IFCandidate,
    article: dict[str, Any],
    config: AppConfig,
    llm_client: RoundRobinLLMClient | None = None,
) -> JudgeResult:
    if not config.enable_judge:
        return heuristic_judge_candidate(candidate, article, config)

    snippets = resolve_evidence_snippets(article, candidate.source_refs)
    if not snippets:
        result = heuristic_judge_candidate(candidate, article, config)
        return result.model_copy(
            update={
                "judge": result.judge.model_copy(
                    update={
                        "verdict": "review",
                        "issues": list(dict.fromkeys(result.judge.issues + ["insufficient_grounding_evidence"])),
                        "needs_human_review": True,
                    }
                )
            }
        )
    if llm_client is None:
        raise ValueError("enable_judge=true requer llm_client")

    payload = llm_client.chat_json(
        [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": render_judge_user_prompt(
                    title=str(article.get("title") or ""),
                    page_id=normalize_page_id(article.get("page_id")),
                    candidate=candidate.model_dump(),
                    evidence_snippets=snippets,
                ),
            },
        ],
        model=config.judge_model_name,
        temperature=0.0,
        max_tokens=1024,
    )
    payload["candidate_id"] = payload.get("candidate_id") or candidate.candidate_id
    payload["source_page_id"] = normalize_page_id(article.get("page_id"))
    payload["source_title"] = str(article.get("title") or "")
    payload["judge_model_name"] = config.judge_model_name
    return JudgeResult.model_validate(payload)

