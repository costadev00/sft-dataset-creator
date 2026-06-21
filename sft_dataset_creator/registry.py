from __future__ import annotations

import importlib
from importlib.metadata import entry_points
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Callable, Protocol, runtime_checkable

from sft_dataset_creator.config import ComponentConfig, SourceConfig
from sft_dataset_creator.models import (
    BatchGenerationResult,
    BackendResponse,
    Document,
    EvaluationResult,
    GenerationRequest,
    SFTCandidate,
)


@runtime_checkable
class SourceAdapter(Protocol):
    def iter_documents(self) -> Any: ...


@runtime_checkable
class TaskRecipe(Protocol):
    name: str

    def build_request(
        self,
        document: Document,
        *,
        slot_id: str,
        difficulty: str,
        max_input_tokens: int,
        token_counter: Callable[[str], int],
    ) -> GenerationRequest: ...

    def candidate_from_response(
        self,
        response: BackendResponse,
        *,
        document: Document,
        slot_id: str,
        attempt: int,
        generator: str,
        model: str,
    ) -> SFTCandidate: ...


@runtime_checkable
class BatchTaskRecipe(Protocol):
    def build_requests(
        self,
        items: Sequence[tuple[Document, str, str]],
        *,
        max_input_tokens: int,
        token_counter_many: Callable[[Sequence[str]], list[int]],
    ) -> list[GenerationRequest]: ...


@runtime_checkable
class GenerationBackend(Protocol):
    model: str

    def count_tokens(self, text: str) -> int: ...

    def generate_json(self, request: GenerationRequest) -> BackendResponse: ...

    def close(self) -> None: ...


@runtime_checkable
class BatchGenerationBackend(Protocol):
    def generate_many(self, requests: Iterable[GenerationRequest]) -> Iterator[BatchGenerationResult]: ...


@runtime_checkable
class BatchTokenCounter(Protocol):
    def count_tokens_many(self, texts: Sequence[str]) -> list[int]: ...


@runtime_checkable
class OutputExporter(Protocol):
    name: str

    def render(self, candidate: SFTCandidate) -> dict[str, Any]: ...


@runtime_checkable
class EvaluationStrategy(Protocol):
    def deterministic(self, candidate: SFTCandidate, document: Document, accepted: Any) -> Any: ...

    def should_route(self, candidate: SFTCandidate, evaluation: Any, seed: int) -> bool: ...

    def llm(self, candidate: SFTCandidate, document: Document, backend: Any) -> Any: ...


@runtime_checkable
class BatchEvaluationStrategy(Protocol):
    def build_llm_request(self, candidate: SFTCandidate, document: Document) -> GenerationRequest: ...

    def evaluation_from_response(self, candidate: SFTCandidate, response: BackendResponse) -> EvaluationResult: ...


Factory = Callable[[Any], Any]
_REGISTRIES: dict[str, dict[str, Factory]] = {
    "sources": {},
    "tasks": {},
    "backends": {},
    "evaluators": {},
    "exporters": {},
}
_ENTRY_POINT_GROUPS = {
    "sources": "sft_dataset_creator.sources",
    "tasks": "sft_dataset_creator.tasks",
    "backends": "sft_dataset_creator.backends",
    "evaluators": "sft_dataset_creator.evaluators",
    "exporters": "sft_dataset_creator.exporters",
}
_BUILTINS_LOADED = False


def _load_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    for module in ("sources", "tasks", "backends", "evaluators", "exporters"):
        importlib.import_module(f"sft_dataset_creator.{module}")
    _BUILTINS_LOADED = True


def register(kind: str, name: str, factory: Factory) -> None:
    if kind not in _REGISTRIES:
        raise KeyError(f"unknown plugin kind: {kind}")
    _REGISTRIES[kind][name] = factory


def _load_entry_points(kind: str) -> None:
    group = _ENTRY_POINT_GROUPS[kind]
    for item in entry_points(group=group):
        _REGISTRIES[kind].setdefault(item.name, item.load())


def create(kind: str, name: str, config: ComponentConfig | SourceConfig | dict[str, Any] | None = None) -> Any:
    _load_builtins()
    if kind not in _REGISTRIES:
        raise KeyError(f"unknown plugin kind: {kind}")
    if name not in _REGISTRIES[kind]:
        _load_entry_points(kind)
    try:
        factory = _REGISTRIES[kind][name]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRIES[kind])) or "none"
        raise KeyError(f"unknown {kind} plugin {name!r}; available: {available}") from exc
    return factory(config)


def available_plugins() -> dict[str, list[str]]:
    _load_builtins()
    for kind in _REGISTRIES:
        _load_entry_points(kind)
    return {kind: sorted(plugins) for kind, plugins in _REGISTRIES.items()}
