from __future__ import annotations

from sft_dataset_creator.models import Document
from sft_dataset_creator.profiles import document_is_eligible


def _document(
    text: str,
    *,
    title: str = "Livro/Capitulo",
    namespace: int = 0,
) -> Document:
    return Document(
        id="doc-1",
        text=text,
        title=title,
        metadata={"ns": namespace},
        source="test",
    )


def test_wikipedia_profile_rejects_non_main_namespace() -> None:
    text = "Este paragrafo tem conteudo suficiente para passar pelo corte minimo. " * 8

    assert not document_is_eligible(_document(text, namespace=102), "wikipedia_ptbr")


def test_wikibooks_profile_does_not_require_main_namespace() -> None:
    text = "Este paragrafo tem conteudo suficiente para passar pelo corte minimo. " * 8

    assert document_is_eligible(_document(text, namespace=102), "wikibooks_ptbr")


def test_wikibooks_profile_rejects_text_smaller_than_a_paragraph() -> None:
    text = "Curto demais para gerar uma tarefa SFT util."

    assert not document_is_eligible(_document(text), "wikibooks_ptbr")


def test_wikibooks_profile_rejects_auxiliary_titles() -> None:
    text = "Este paragrafo tem conteudo suficiente para passar pelo corte minimo. " * 8

    assert not document_is_eligible(_document(text, title="Wikilivros:Ajuda"), "wikibooks_ptbr")
