from __future__ import annotations

import pytest

from sft_dataset_creator.models import Document
from sft_dataset_creator.tasks import CHAT_TEMPLATE_TOKEN_MARGIN, GenericTaskRecipe


def test_generation_context_reserves_full_prompt_overhead() -> None:
    recipe = GenericTaskRecipe("summarization", language="pt-BR")
    request = recipe.build_request(
        Document(id="doc", title="Long document", text="A" * 5_000, source="test"),
        slot_id="slot-1",
        difficulty="medium",
        max_input_tokens=1_200,
        token_counter=len,
    )

    rendered_prompt = "\n".join(message.content for message in request.messages)
    assert len(rendered_prompt) <= 1_200 - CHAT_TEMPLATE_TOKEN_MARGIN
    assert request.metadata["context_truncated"] is True


def test_generation_rejects_budget_smaller_than_prompt_overhead() -> None:
    recipe = GenericTaskRecipe("closed_qa")
    with pytest.raises(ValueError, match="too small for the generation prompt"):
        recipe.build_request(
            Document(id="doc", text="Grounded source text.", source="test"),
            slot_id="slot-1",
            difficulty="easy",
            max_input_tokens=32,
            token_counter=len,
        )
