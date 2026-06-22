from __future__ import annotations

import hashlib
import json
from typing import Iterable

from sft_dataset_creator.backends import BackendProcess
from sft_dataset_creator.config import EvaluationConfig
from sft_dataset_creator.models import (
    BackendResponse,
    ChatMessage,
    Document,
    EvaluationResult,
    GenerationRequest,
    SFTCandidate,
)
from sft_dataset_creator.prompts import JUDGE_CRITERIA, JUDGE_SYSTEM_PROMPT
from sft_dataset_creator.registry import register
from sft_dataset_creator.quality import candidate_fingerprint, candidate_source_reference


JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "overall_score", "scores", "issues"],
    "properties": {
        "verdict": {"type": "string", "enum": ["accept", "reject", "review"]},
        "overall_score": {"type": "number", "minimum": 0, "maximum": 5},
        "scores": {
            "type": "object",
            "required": ["grounding", "instruction_quality", "output_quality"],
            "properties": {
                "grounding": {"type": "number", "minimum": 0, "maximum": 5},
                "instruction_quality": {"type": "number", "minimum": 0, "maximum": 5},
                "output_quality": {"type": "number", "minimum": 0, "maximum": 5},
            },
        },
        "issues": {"type": "array", "items": {"type": "string"}},
    },
}


def _evidence_text(document: Document, candidate: SFTCandidate) -> tuple[list[str], list[str]]:
    sections = {section.id: section.text for section in document.evidence_sections()}
    snippets: list[str] = []
    issues: list[str] = []
    for evidence in candidate.evidence:
        section = sections.get(evidence.section_id)
        if section is None:
            issues.append("invalid_section")
            continue
        if evidence.end > len(section) or evidence.start >= evidence.end:
            issues.append("invalid_evidence_offsets")
            continue
        quote = section[evidence.start : evidence.end]
        if evidence.quote and evidence.quote.strip() != quote.strip():
            issues.append("evidence_quote_mismatch")
            continue
        if len(quote.strip()) < 20:
            issues.append("weak_grounding")
        snippets.append(quote)
    if not snippets:
        issues.append("missing_grounding")
    return snippets, list(dict.fromkeys(issues))


def deterministic_evaluation(
    candidate: SFTCandidate,
    document: Document,
    accepted: Iterable[SFTCandidate],
    config: EvaluationConfig,
    accepted_fingerprints: set[str] | None = None,
) -> EvaluationResult:
    issues: list[str] = []
    if not candidate.instruction.strip() or not candidate.output.strip():
        issues.append("empty_instruction_or_output")
    if len(candidate.instruction) < 12:
        issues.append("instruction_too_short")
    if candidate_source_reference(candidate.instruction, candidate.input, candidate.output):
        issues.append("source_reference_in_candidate")
    fingerprint = candidate_fingerprint(candidate.instruction, candidate.input, candidate.output)
    if accepted_fingerprints is None:
        accepted_fingerprints = {
            candidate_fingerprint(item.instruction, item.input, item.output) for item in accepted
        }
    if fingerprint in accepted_fingerprints:
        issues.append("duplicate_candidate")
    snippets, evidence_issues = _evidence_text(document, candidate)
    issues.extend(evidence_issues)
    critical = {
        "empty_instruction_or_output",
        "invalid_section",
        "invalid_evidence_offsets",
        "evidence_quote_mismatch",
        "missing_grounding",
        "source_reference_in_candidate",
        "instruction_too_short",
        "duplicate_candidate",
    }
    verdict = "reject" if critical.intersection(issues) else "accept"
    return EvaluationResult(
        candidate_id=candidate.id,
        verdict=verdict,
        evaluator="deterministic",
        overall_score=5.0 if verdict == "accept" else 0.0,
        scores={"grounding": 5.0 if snippets else 0.0},
        issues=list(dict.fromkeys(issues)),
    )


def should_route_to_llm(
    candidate: SFTCandidate,
    evaluation: EvaluationResult,
    config: EvaluationConfig,
    seed: int,
) -> bool:
    if evaluation.verdict != "accept" or config.llm is None:
        return False
    routing = config.routing
    if routing.judge_hard_tasks and candidate.difficulty == "hard":
        return True
    if routing.judge_truncated_context and candidate.metadata.get("context_truncated"):
        return True
    if routing.judge_high_risk and candidate.metadata.get("risk") == "high":
        return True
    if routing.judge_weak_grounding and "weak_grounding" in evaluation.issues:
        return True
    digest = hashlib.sha256(f"{seed}:{candidate.id}".encode("utf-8")).digest()
    sample = int.from_bytes(digest[:8], "big") / 2**64
    return sample < routing.audit_fraction


def llm_evaluation(
    candidate: SFTCandidate,
    document: Document,
    backend: BackendProcess,
    config: EvaluationConfig,
) -> EvaluationResult:
    request = build_llm_request(candidate, document, config)
    response = backend.generate_json(request)
    return evaluation_from_response(candidate, response, config)


def build_llm_request(
    candidate: SFTCandidate,
    document: Document,
    config: EvaluationConfig,
) -> GenerationRequest:
    snippets, _issues = _evidence_text(document, candidate)
    payload = {
        "candidate": {
            "task": candidate.task,
            "difficulty": candidate.difficulty,
            "instruction": candidate.instruction,
            "input": candidate.input,
            "output": candidate.output,
        },
        "evidence": snippets,
        "criteria": JUDGE_CRITERIA,
    }
    request = GenerationRequest(
        slot_id=candidate.slot_id,
        document_id=document.id,
        task="judge",
        difficulty=candidate.difficulty,
        messages=[
            ChatMessage(role="system", content=JUDGE_SYSTEM_PROMPT),
            ChatMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
        ],
        response_schema=JUDGE_SCHEMA,
        max_output_tokens=min(2048, config.llm.max_output_tokens if config.llm else 2048),
    )
    return request


def evaluation_from_response(
    candidate: SFTCandidate,
    response: BackendResponse,
    config: EvaluationConfig,
) -> EvaluationResult:
    value = response.payload
    scores = {key: float(score) for key, score in dict(value.get("scores") or {}).items()}
    overall = float(value.get("overall_score", 0.0))
    verdict = str(value.get("verdict") or "review")
    if verdict == "accept" and (
        overall < config.acceptance.overall_score
        or scores.get("grounding", 0.0) < config.acceptance.grounding_score
    ):
        verdict = "review"
    if verdict not in {"accept", "reject", "review"}:
        verdict = "review"
    return EvaluationResult(
        candidate_id=candidate.id,
        verdict=verdict,
        evaluator=f"llm:{response.model}",
        selected_for_llm=True,
        overall_score=overall,
        scores=scores,
        issues=[str(item) for item in value.get("issues", [])],
    )


class CompositeEvaluator:
    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config
        self._accepted_ids: set[str] = set()
        self._accepted_fingerprints: set[str] = set()

    def _sync_accepted(self, accepted: Iterable[SFTCandidate]) -> None:
        for item in accepted:
            if item.id in self._accepted_ids:
                continue
            self._accepted_ids.add(item.id)
            self._accepted_fingerprints.add(
                candidate_fingerprint(item.instruction, item.input, item.output)
            )

    def deterministic(
        self,
        candidate: SFTCandidate,
        document: Document,
        accepted: Iterable[SFTCandidate],
    ) -> EvaluationResult:
        self._sync_accepted(accepted)
        return deterministic_evaluation(
            candidate,
            document,
            accepted,
            self.config,
            accepted_fingerprints=self._accepted_fingerprints,
        )

    def should_route(self, candidate: SFTCandidate, evaluation: EvaluationResult, seed: int) -> bool:
        return should_route_to_llm(candidate, evaluation, self.config, seed)

    def llm(self, candidate: SFTCandidate, document: Document, backend: BackendProcess) -> EvaluationResult:
        return llm_evaluation(candidate, document, backend, self.config)

    def build_llm_request(self, candidate: SFTCandidate, document: Document) -> GenerationRequest:
        return build_llm_request(candidate, document, self.config)

    def evaluation_from_response(self, candidate: SFTCandidate, response: BackendResponse) -> EvaluationResult:
        return evaluation_from_response(candidate, response, self.config)


register("evaluators", "composite", lambda config: CompositeEvaluator(EvaluationConfig.model_validate(config)))
