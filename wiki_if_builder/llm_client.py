from __future__ import annotations

import threading
from typing import Any

from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from wiki_if_builder.utils import parse_json_maybe_markdown


class LLMJSONParseError(ValueError):
    def __init__(self, message: str, raw_response: str, base_url: str) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.base_url = base_url


class RoundRobinLLMClient:
    def __init__(
        self,
        *,
        base_urls: list[str],
        api_key: str,
        model: str,
        timeout_seconds: float = 180.0,
        max_concurrent_calls: int = 1,
    ) -> None:
        if not base_urls:
            raise ValueError("Pelo menos um endpoint OpenAI-compatible deve ser configurado")
        self.base_urls = [url.rstrip("/") for url in base_urls]
        self.model = model
        self._clients = [
            OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_seconds)
            for base_url in self.base_urls
        ]
        self._lock = threading.Lock()
        self._next_index = 0
        self._semaphore = threading.BoundedSemaphore(max(1, max_concurrent_calls))

    @property
    def endpoint_count(self) -> int:
        return len(self.base_urls)

    def next_base_url(self) -> str:
        with self._lock:
            base_url = self.base_urls[self._next_index]
            self._next_index = (self._next_index + 1) % len(self.base_urls)
            return base_url

    def _next_client(self) -> tuple[OpenAI, str]:
        with self._lock:
            idx = self._next_index
            self._next_index = (self._next_index + 1) % len(self._clients)
        return self._clients[idx], self.base_urls[idx]

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=12),
        reraise=True,
    )
    def _chat_raw(
        self,
        *,
        client: OpenAI,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Resposta do LLM veio sem content")
        return content

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        with self._semaphore:
            client, base_url = self._next_client()
            raw = self._chat_raw(
                client=client,
                model=model or self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        try:
            return parse_json_maybe_markdown(raw)
        except Exception as exc:  # noqa: BLE001 - preserva resposta crua para auditoria
            raise LLMJSONParseError(str(exc), raw_response=raw, base_url=base_url) from exc


def build_llm_client(config, *, model: str | None = None) -> RoundRobinLLMClient:
    return RoundRobinLLMClient(
        base_urls=config.effective_base_urls,
        api_key=config.openai_api_key,
        model=model or config.model_name,
        timeout_seconds=config.llm_timeout_seconds,
        max_concurrent_calls=config.resolved_max_concurrent_llm_calls,
    )

