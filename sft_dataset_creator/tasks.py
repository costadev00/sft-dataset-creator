from __future__ import annotations

import json
from collections.abc import Callable, Sequence

from sft_dataset_creator.models import (
    BackendResponse,
    ChatMessage,
    Document,
    EvidenceSpan,
    GenerationRequest,
    SFTCandidate,
)
from sft_dataset_creator.prompts import GENERATOR_SYSTEM_PROMPT, TASK_INSTRUCTIONS
from sft_dataset_creator.registry import register


CANDIDATE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["instruction", "input", "output", "evidence", "risk"],
    "properties": {
        "instruction": {"type": "string"},
        "input": {"type": "string"},
        "output": {"type": "string"},
        "risk": {"type": "string", "enum": ["low", "medium", "high"]},
        "evidence": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["section_id", "start", "end"],
                "properties": {
                    "section_id": {"type": "string"},
                    "start": {"type": "integer", "minimum": 0},
                    "end": {"type": "integer", "minimum": 1},
                    "quote": {"type": "string"},
                },
            },
        },
    },
}


def _render_context(document: Document, content_characters: int | None = None) -> str:
    remaining = content_characters
    blocks: list[str] = []
    for section in document.evidence_sections():
        header = f"[section_id={section.id}]\n"
        if remaining is None:
            blocks.append(header + section.text)
            continue
        if remaining <= 0:
            break
        take = min(remaining, len(section.text))
        if take:
            blocks.append(header + section.text[:take])
        remaining -= take
        if take < len(section.text):
            break
    return "\n\n".join(blocks)


def _contexts(
    documents: Sequence[Document],
    max_tokens: int,
    counter_many: Callable[[Sequence[str]], list[int]],
) -> list[tuple[str, bool]]:
    full = [_render_context(document) for document in documents]
    counts = counter_many(full)
    results: list[tuple[str, bool] | None] = [None] * len(documents)
    active: list[int] = []
    lows: dict[int, int] = {}
    highs: dict[int, int] = {}
    for index, (document, text, count) in enumerate(zip(documents, full, counts, strict=True)):
        if count <= max_tokens:
            results[index] = (text, False)
            continue
        active.append(index)
        lows[index] = 0
        highs[index] = sum(len(section.text) for section in document.evidence_sections())
    while active:
        probes: list[str] = []
        middles: list[int] = []
        for index in active:
            middle = (lows[index] + highs[index] + 1) // 2
            middles.append(middle)
            probes.append(_render_context(documents[index], middle))
        probe_counts = counter_many(probes)
        next_active: list[int] = []
        for index, middle, count in zip(active, middles, probe_counts, strict=True):
            if count <= max_tokens:
                lows[index] = middle
            else:
                highs[index] = middle - 1
            if lows[index] < highs[index]:
                next_active.append(index)
        active = next_active
    for index, value in enumerate(results):
        if value is None:
            results[index] = (_render_context(documents[index], lows[index]), True)
    return [value for value in results if value is not None]


def _context(document: Document, max_tokens: int, counter: Callable[[str], int]) -> tuple[str, bool]:
    return _contexts([document], max_tokens, lambda texts: [counter(text) for text in texts])[0]


class GenericTaskRecipe:
    def __init__(self, name: str, language: str = "en") -> None:
        if name not in TASK_INSTRUCTIONS:
            raise ValueError(f"unknown built-in task: {name}")
        self.name = name
        self.language = language

    def build_request(
        self,
        document: Document,
        *,
        slot_id: str,
        difficulty: str,
        max_input_tokens: int,
        token_counter: Callable[[str], int],
    ) -> GenerationRequest:
        context, truncated = _context(document, max_input_tokens, token_counter)
        return self._request(document, slot_id, difficulty, context, truncated)

    def build_requests(
        self,
        items: Sequence[tuple[Document, str, str]],
        *,
        max_input_tokens: int,
        token_counter_many: Callable[[Sequence[str]], list[int]],
    ) -> list[GenerationRequest]:
        contexts = _contexts(
            [document for document, _slot_id, _difficulty in items],
            max_input_tokens,
            token_counter_many,
        )
        return [
            self._request(document, slot_id, difficulty, context, truncated)
            for (document, slot_id, difficulty), (context, truncated) in zip(items, contexts, strict=True)
        ]

    def _request(
        self,
        document: Document,
        slot_id: str,
        difficulty: str,
        context: str,
        truncated: bool,
    ) -> GenerationRequest:
        user = {
            "task": self.name,
            "language": self.language,
            "difficulty": difficulty,
            "task_guidance": TASK_INSTRUCTIONS[self.name],
            "document_id": document.id,
            "title": document.title,
            "context_truncated": truncated,
            "source_excerpt": context,
        }
        return GenerationRequest(
            slot_id=slot_id,
            document_id=document.id,
            task=self.name,
            difficulty=difficulty,
            messages=[
                ChatMessage(role="system", content=GENERATOR_SYSTEM_PROMPT),
                ChatMessage(role="user", content=json.dumps(user, ensure_ascii=False)),
            ],
            response_schema=CANDIDATE_SCHEMA,
            max_output_tokens=4096,
            metadata={"context_truncated": truncated},
        )

    def candidate_from_response(
        self,
        response: BackendResponse,
        *,
        document: Document,
        slot_id: str,
        attempt: int,
        generator: str,
        model: str,
    ) -> SFTCandidate:
        payload = response.payload
        evidence = [
            EvidenceSpan(
                document_id=document.id,
                section_id=str(item.get("section_id", "0")),
                start=int(item.get("start", 0)),
                end=int(item.get("end", 0)),
                quote=str(item["quote"]) if item.get("quote") else None,
            )
            for item in payload.get("evidence", [])
        ]
        instruction = str(payload.get("instruction") or "").strip()
        input_text = str(payload.get("input") or "").strip()
        output = str(payload.get("output") or "").strip()
        user_content = instruction if not input_text else f"{instruction}\n\nContext:\n{input_text}"
        return SFTCandidate(
            id=f"{slot_id}-a{attempt}",
            slot_id=slot_id,
            attempt=attempt,
            source=document.source,
            document_id=document.id,
            source_title=document.title,
            task=self.name,
            difficulty=str(payload.get("difficulty") or "medium"),
            instruction=instruction,
            input=input_text,
            output=output,
            messages=[
                ChatMessage(role="user", content=user_content),
                ChatMessage(role="assistant", content=output),
            ],
            evidence=evidence,
            metadata={
                "risk": str(payload.get("risk") or "medium"),
                "raw_response_stored": response.raw_text is not None,
            },
            generator=generator,
            model=model,
        )


for _task_name in TASK_INSTRUCTIONS:
    register(
        "tasks",
        _task_name,
        lambda config, name=_task_name: GenericTaskRecipe(
            name,
            language=str(config.get("language", "en")) if isinstance(config, dict) else "en",
        ),
    )
