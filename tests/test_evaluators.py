from __future__ import annotations

from sft_dataset_creator.config import EvaluationConfig, GenerationConfig
from sft_dataset_creator.evaluators import deterministic_evaluation, should_route_to_llm
from sft_dataset_creator.models import ChatMessage, Document, EvidenceSpan, SFTCandidate


def _candidate(**updates) -> SFTCandidate:
    values = {
        "id": "slot-1-a1",
        "slot_id": "slot-1",
        "attempt": 1,
        "source": "test",
        "document_id": "doc",
        "task": "closed_qa",
        "difficulty": "hard",
        "instruction": "Answer this grounded question.",
        "output": "A factual answer.",
        "messages": [ChatMessage(role="user", content="Question"), ChatMessage(role="assistant", content="Answer")],
        "evidence": [EvidenceSpan(document_id="doc", section_id="0", start=0, end=24)],
        "generator": "fake",
        "model": "fake",
    }
    values.update(updates)
    return SFTCandidate(**values)


def test_invalid_evidence_is_rejected() -> None:
    document = Document(id="doc", text="short", source="test")
    result = deterministic_evaluation(_candidate(), document, [], EvaluationConfig())
    assert result.verdict == "reject"
    assert "invalid_evidence_offsets" in result.issues


def test_hard_candidate_routes_when_llm_is_configured() -> None:
    document = Document(id="doc", text="A sufficiently long grounded document passage.", source="test")
    config = EvaluationConfig(llm=GenerationConfig(plugin="fake", model="judge"))
    candidate = _candidate()
    result = deterministic_evaluation(candidate, document, [], config)
    assert result.verdict == "accept"
    assert should_route_to_llm(candidate, result, config, seed=42)


def test_source_dependent_instruction_is_rejected() -> None:
    document = Document(id="doc", text="A sufficiently long grounded document passage.", source="test")
    candidate = _candidate(instruction="De acordo com o texto, qual é a resposta factual?")

    result = deterministic_evaluation(candidate, document, [], EvaluationConfig())

    assert result.verdict == "reject"
    assert "source_reference_in_candidate" in result.issues


def test_source_reference_in_output_is_rejected() -> None:
    document = Document(id="doc", text="A sufficiently long grounded document passage.", source="test")
    candidate = _candidate(output="De acordo com o texto, a resposta factual é esta.")

    result = deterministic_evaluation(candidate, document, [], EvaluationConfig())

    assert result.verdict == "reject"
    assert "source_reference_in_candidate" in result.issues


def test_subject_matter_use_of_context_is_not_rejected() -> None:
    document = Document(id="doc", text="A sufficiently long grounded document passage.", source="test")
    candidate = _candidate(instruction="Explique o papel jurídico do contexto metropolitano.")

    result = deterministic_evaluation(candidate, document, [], EvaluationConfig())

    assert result.verdict == "accept"
    assert "source_reference_in_candidate" not in result.issues
