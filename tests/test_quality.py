from __future__ import annotations

import pytest

from sft_dataset_creator.quality import candidate_source_reference, source_reference


@pytest.mark.parametrize(
    "instruction",
    [
        "De acordo com o texto, qual é a resposta?",
        "Explique o resultado com base no documento fornecido.",
        "Qual é o conceito mencionado no trecho?",
        "Resuma a literatura citada no texto.",
        "According to the passage, what happened?",
        "Segun el documento, que ocurrio?",
    ],
)
def test_source_reference_detects_hidden_context(instruction: str) -> None:
    assert source_reference(instruction) is not None


@pytest.mark.parametrize(
    "instruction",
    [
        "Explique o contexto metropolitano de Belo Horizonte.",
        "Quais direitos constam no texto constitucional brasileiro?",
        "Compare fontes primárias e secundárias em historiografia.",
        "Segundo a Real Academia Espanhola, qual é a origem do termo?",
    ],
)
def test_source_reference_allows_self_contained_subjects(instruction: str) -> None:
    assert source_reference(instruction) is None


def test_candidate_source_reference_checks_every_user_facing_field() -> None:
    assert candidate_source_reference("Pergunta direta.", "", "According to the document, yes.") is not None
