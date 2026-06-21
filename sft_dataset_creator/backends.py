from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import multiprocessing
import os
import re
import threading
import time
import traceback
import uuid
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from sft_dataset_creator.config import GenerationConfig
from sft_dataset_creator.models import BatchGenerationResult, BackendResponse, GenerationRequest
from sft_dataset_creator.registry import create, register


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model response is not a JSON object")
    return value


class ApproximateTokenCounter:
    def count_tokens(self, text: str) -> int:
        return max(1, (len(text) + 3) // 4)

    def count_tokens_many(self, texts: Sequence[str]) -> list[int]:
        return [self.count_tokens(text) for text in texts]


class HuggingFaceTokenCounter:
    def __init__(self, config: GenerationConfig) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError("Exact local token counting requires transformers") from exc
        params = config.params
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(params.get("tokenizer") or config.model),
            revision=params.get("tokenizer_revision") or config.model_revision,
            trust_remote_code=bool(params.get("trust_remote_code", False)),
            cache_dir=params.get("download_dir"),
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def count_tokens_many(self, texts: Sequence[str]) -> list[int]:
        if not texts:
            return []
        encoded = self.tokenizer(list(texts), add_special_tokens=False, return_length=True)
        lengths = encoded.get("length")
        if lengths is not None:
            return [int(length) for length in lengths]
        return [len(token_ids) for token_ids in encoded["input_ids"]]


def create_token_counter(config: GenerationConfig) -> ApproximateTokenCounter | HuggingFaceTokenCounter:
    if config.plugin == "vllm_local":
        return HuggingFaceTokenCounter(config)
    return ApproximateTokenCounter()


class FakeBackend:
    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self.model = config.model

    def count_tokens(self, text: str) -> int:
        return max(1, (len(text) + 3) // 4)

    def count_tokens_many(self, texts: Sequence[str]) -> list[int]:
        return [self.count_tokens(text) for text in texts]

    def _delay(self, request: GenerationRequest) -> float:
        delays = self.config.params.get("delay_by_request") or {}
        return float(delays.get(request.request_id, 0.0)) if isinstance(delays, dict) else 0.0

    def _response(self, request: GenerationRequest) -> BackendResponse:
        if request.task == "judge":
            payload = {
                "verdict": "accept",
                "overall_score": 4.8,
                "scores": {"grounding": 5.0, "instruction_quality": 4.5, "output_quality": 4.8},
                "issues": [],
            }
        else:
            marker = hashlib.sha256(f"{request.slot_id}:{request.document_id}".encode("utf-8")).hexdigest()[:20]
            payload = {
                "instruction": f"Complete this {request.task} task at {request.difficulty} difficulty.",
                "input": "",
                "output": (
                    f"A grounded {request.task} answer for {request.document_id}. "
                    f"Deterministic test marker: {marker}."
                ),
                "risk": "low",
                "evidence": [
                    {"section_id": str(request.metadata.get("chunk_id") or "0"), "start": 0, "end": 24}
                ],
            }
        return BackendResponse(payload=payload, backend="fake", model=self.model)

    def generate_json(self, request: GenerationRequest) -> BackendResponse:
        delay = self._delay(request)
        if delay:
            time.sleep(delay)
        return self._response(request)

    async def async_generate_json(self, request: GenerationRequest) -> BackendResponse:
        delay = self._delay(request)
        if delay:
            await asyncio.sleep(delay)
        return self._response(request)

    def close(self) -> None:
        return None


class OpenAICompatibleBackend:
    def __init__(self, config: GenerationConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("OpenAI-compatible backends require the 'openai' extra") from exc
        self.config = config
        self.model = config.model
        params = config.params
        api_key = os.getenv(str(params.get("api_key_env") or "OPENAI_API_KEY"), "local-token")
        self.client = OpenAI(
            base_url=params.get("base_url"),
            api_key=api_key,
            timeout=float(params.get("timeout", 180)),
        )

    def count_tokens(self, text: str) -> int:
        return max(1, (len(text) + 3) // 4)

    def count_tokens_many(self, texts: Sequence[str]) -> list[int]:
        return [self.count_tokens(text) for text in texts]

    def generate_json(self, request: GenerationRequest) -> BackendResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [message.model_dump() for message in request.messages],
            "temperature": self.config.temperature,
            "max_tokens": request.max_output_tokens,
        }
        if request.seed is not None:
            kwargs["seed"] = request.seed
        try:
            response = self.client.chat.completions.create(
                **kwargs,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "response", "strict": True, "schema": request.response_schema},
                },
            )
        except TypeError:
            kwargs.pop("seed", None)
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            message = str(exc).lower()
            if status_code == 400 and ("response_format" in message or "json_schema" in message):
                response = self.client.chat.completions.create(**kwargs)
            else:
                raise
        raw = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        return BackendResponse(
            payload=_parse_json(raw),
            raw_text=raw,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            backend="openai_compatible",
            model=self.model,
        )

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if close:
            close()


class VLLMLocalBackend:
    def __init__(self, config: GenerationConfig) -> None:
        try:
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError("Local inference requires the 'local' extra with vLLM") from exc
        self.config = config
        self.model = config.model
        self._sampling_cls = SamplingParams
        params = dict(config.params)
        self.enable_thinking = bool(params.pop("enable_thinking", False))
        self.async_mode = config.batching.mode != "sequential"
        tokenizer_name = str(params.get("tokenizer") or config.model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            revision=params.get("tokenizer_revision") or config.model_revision,
            trust_remote_code=bool(params.get("trust_remote_code", False)),
            cache_dir=params.get("download_dir"),
        )
        llm_kwargs: dict[str, Any] = {
            "model": config.model,
            "revision": config.model_revision,
            "tensor_parallel_size": int(params.pop("tensor_parallel_size", 1)),
            "gpu_memory_utilization": float(params.pop("gpu_memory_utilization", 0.92)),
            "max_model_len": config.context_window,
            "dtype": params.pop("dtype", "auto"),
            "trust_remote_code": bool(params.pop("trust_remote_code", False)),
        }
        forwarded = (
            "download_dir",
            "tokenizer",
            "tokenizer_revision",
            "quantization",
            "enforce_eager",
            "kv_cache_dtype",
            "max_num_seqs",
            "max_num_batched_tokens",
            "enable_chunked_prefill",
            "enable_prefix_caching",
            "disable_log_stats",
            "cpu_offload_gb",
            "kv_cache_memory_bytes",
        )
        for key in forwarded:
            if key in params and params[key] is not None:
                llm_kwargs[key] = params.pop(key)
        if llm_kwargs["revision"] is None:
            llm_kwargs.pop("revision")
        if self.async_mode:
            try:
                from vllm.engine.arg_utils import AsyncEngineArgs
                from vllm.v1.engine.async_llm import AsyncLLM
            except ImportError as exc:
                if config.batching.mode == "async":
                    raise ImportError("vLLM 0.23 AsyncLLM is required for async batching") from exc
                self.async_mode = False
            else:
                self.llm = AsyncLLM.from_engine_args(AsyncEngineArgs(**llm_kwargs))
        if not self.async_mode:
            self.llm = LLM(**llm_kwargs)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def count_tokens_many(self, texts: Sequence[str]) -> list[int]:
        if not texts:
            return []
        encoded = self.tokenizer(list(texts), add_special_tokens=False, return_length=True)
        lengths = encoded.get("length")
        if lengths is not None:
            return [int(length) for length in lengths]
        return [len(token_ids) for token_ids in encoded["input_ids"]]

    def _sampling(self, request: GenerationRequest):
        kwargs: dict[str, Any] = {
            "temperature": self.config.temperature,
            "max_tokens": request.max_output_tokens,
        }
        if request.seed is not None:
            kwargs["seed"] = request.seed
        try:
            from vllm.sampling_params import StructuredOutputsParams

            kwargs["structured_outputs"] = StructuredOutputsParams(json=request.response_schema)
        except (ImportError, TypeError, ValueError):
            pass
        return self._sampling_cls(**kwargs)

    def _messages(self, request: GenerationRequest) -> list[dict[str, Any]]:
        return [message.model_dump() for message in request.messages]

    def _prompt(self, request: GenerationRequest) -> str:
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        try:
            return str(
                self.tokenizer.apply_chat_template(
                    self._messages(request),
                    **kwargs,
                    enable_thinking=self.enable_thinking,
                )
            )
        except TypeError:
            return str(self.tokenizer.apply_chat_template(self._messages(request), **kwargs))

    @staticmethod
    def _response_from_output(output: Any, model: str) -> BackendResponse:
        if output is None or not output.outputs:
            raise RuntimeError("vLLM returned no output")
        raw = output.outputs[0].text
        return BackendResponse(
            payload=_parse_json(raw),
            raw_text=raw,
            input_tokens=len(output.prompt_token_ids or []),
            output_tokens=len(output.outputs[0].token_ids or []),
            backend="vllm_local",
            model=model,
        )

    def generate_json(self, request: GenerationRequest) -> BackendResponse:
        if self.async_mode:
            raise RuntimeError("async vLLM requests must be submitted through BackendProcess.generate_many")
        messages = self._messages(request)
        sampling = self._sampling(request)
        try:
            outputs = self.llm.chat(
                messages,
                sampling_params=sampling,
                chat_template_kwargs={"enable_thinking": self.enable_thinking},
            )
        except TypeError:
            outputs = self.llm.chat(messages, sampling_params=sampling)
        return self._response_from_output(outputs[0] if outputs else None, self.model)

    async def async_generate_json(self, request: GenerationRequest) -> BackendResponse:
        if not self.async_mode:
            return await asyncio.to_thread(self.generate_json, request)
        final_output = None
        request_id = request.request_id or uuid.uuid4().hex
        async for output in self.llm.generate(self._prompt(request), self._sampling(request), request_id):
            final_output = output
        return self._response_from_output(final_output, self.model)

    async def async_close(self) -> None:
        shutdown = getattr(self.llm, "shutdown", None)
        if shutdown:
            result = shutdown()
            if inspect.isawaitable(result):
                await result

    def close(self) -> None:
        shutdown = getattr(self.llm, "shutdown", None)
        if shutdown and not self.async_mode:
            shutdown()


async def _close_backend(backend: Any) -> None:
    async_close = getattr(backend, "async_close", None)
    if async_close is not None:
        await async_close()
        return
    close = getattr(backend, "close", None)
    if close is not None:
        close()


async def _generate_one(backend: Any, request: GenerationRequest, use_async: bool) -> BackendResponse:
    async_generate = getattr(backend, "async_generate_json", None)
    if use_async and async_generate is not None:
        return await async_generate(request)
    return backend.generate_json(request)


async def _connection_recv(connection: Any) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[dict[str, Any]] = loop.create_future()
    descriptor = connection.fileno()

    def receive() -> None:
        loop.remove_reader(descriptor)
        try:
            future.set_result(connection.recv())
        except BaseException as exc:
            future.set_exception(exc)

    loop.add_reader(descriptor, receive)
    try:
        return await future
    finally:
        loop.remove_reader(descriptor)


async def _process_batch(connection: Any, backend: Any, config: GenerationConfig, batch_id: str) -> None:
    use_async = config.batching.mode != "sequential" and hasattr(backend, "async_generate_json")
    concurrency = config.batching.max_inflight_requests if use_async else 1
    pending: dict[asyncio.Task[BackendResponse], tuple[str, float]] = {}
    ended = False
    submitted = 0
    completed = 0

    while not ended or pending:
        while not ended and len(pending) < concurrency:
            command = await _connection_recv(connection)
            if command.get("batch_id") != batch_id:
                raise RuntimeError("backend command stream became desynchronized")
            if command["kind"] == "batch_end":
                ended = True
                break
            if command["kind"] != "batch_item":
                raise ValueError(f"unexpected command during batch: {command['kind']}")
            request = GenerationRequest.model_validate(command["request"])
            request_id = request.request_id or f"request-{submitted:08d}"
            if request.request_id is None:
                request = request.model_copy(update={"request_id": request_id})
            task = asyncio.create_task(_generate_one(backend, request, use_async))
            pending[task] = (request_id, time.perf_counter())
            submitted += 1
        if not pending:
            continue
        done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            request_id, started = pending.pop(task)
            latency = time.perf_counter() - started
            try:
                response = task.result()
                result = BatchGenerationResult(
                    request_id=request_id,
                    response=response,
                    latency_seconds=latency,
                    queue_depth=len(pending),
                )
            except BaseException as exc:
                result = BatchGenerationResult(
                    request_id=request_id,
                    error=f"{type(exc).__name__}: {exc}",
                    latency_seconds=latency,
                    queue_depth=len(pending),
                )
            completed += 1
            connection.send({"kind": "batch_result", "batch_id": batch_id, "result": result.model_dump(mode="json")})
    connection.send({"kind": "batch_done", "batch_id": batch_id, "submitted": submitted, "completed": completed})


async def _worker_async(connection: Any, config_payload: dict[str, Any]) -> None:
    backend = None
    try:
        config = GenerationConfig.model_validate(config_payload)
        backend = create("backends", config.plugin, config)
        connection.send({"kind": "ready", "ok": True, "model": backend.model})
        while True:
            command = await _connection_recv(connection)
            kind = command["kind"]
            if kind == "close":
                break
            if kind == "count_tokens":
                count = backend.count_tokens(command["text"])
                connection.send({"kind": "call_result", "call_id": command["call_id"], "ok": True, "result": count})
                continue
            if kind == "count_tokens_many":
                counter = getattr(backend, "count_tokens_many", None)
                if counter is None:
                    counts = [backend.count_tokens(text) for text in command["texts"]]
                else:
                    counts = counter(command["texts"])
                connection.send(
                    {"kind": "call_result", "call_id": command["call_id"], "ok": True, "result": counts}
                )
                continue
            if kind == "batch_start":
                await _process_batch(connection, backend, config, command["batch_id"])
                continue
            raise ValueError(f"unknown backend command: {kind}")
    except BaseException as exc:
        try:
            connection.send({"kind": "fatal", "ok": False, "error": str(exc), "traceback": traceback.format_exc()})
        except (BrokenPipeError, EOFError):
            pass
    finally:
        if backend is not None:
            try:
                await _close_backend(backend)
            except Exception:
                pass
        connection.close()


def _worker(connection: Any, config_payload: dict[str, Any]) -> None:
    asyncio.run(_worker_async(connection, config_payload))


class BackendProcess:
    def __init__(self, config: GenerationConfig, startup_timeout: float = 1800.0) -> None:
        self.config = config
        self.startup_timeout = startup_timeout
        self.process: multiprocessing.Process | None = None
        self.connection: Any = None
        self.startup_seconds = 0.0
        self._lock = threading.Lock()

    def __enter__(self) -> "BackendProcess":
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        self.process = context.Process(
            target=_worker,
            args=(child, self.config.model_dump(mode="json")),
        )
        started = time.perf_counter()
        self.process.start()
        self.connection = parent
        message = self._get_result(self.startup_timeout)
        self.startup_seconds = time.perf_counter() - started
        self._raise_if_failed(message)
        if message.get("kind") != "ready":
            raise RuntimeError(f"unexpected backend startup message: {message.get('kind')}")
        return self

    def _get_result(self, timeout: float | None = None) -> dict[str, Any]:
        if self.connection is None:
            raise RuntimeError("backend process is not running")
        if not self.connection.poll(timeout):
            if self.process is not None and not self.process.is_alive():
                raise RuntimeError(f"backend process exited with code {self.process.exitcode}")
            raise TimeoutError("timed out waiting for backend process")
        try:
            return self.connection.recv()
        except EOFError as exc:
            raise RuntimeError("backend process closed its connection") from exc

    @staticmethod
    def _raise_if_failed(message: dict[str, Any]) -> None:
        if message.get("kind") == "fatal" or message.get("ok") is False:
            raise RuntimeError(f"backend process failed: {message.get('error')}\n{message.get('traceback', '')}")

    def _call(self, kind: str, **payload: Any) -> Any:
        if self.connection is None:
            raise RuntimeError("backend process is not running")
        call_id = uuid.uuid4().hex
        with self._lock:
            self.connection.send({"kind": kind, "call_id": call_id, **payload})
            message = self._get_result(self.config.batching.request_timeout_seconds)
            self._raise_if_failed(message)
            if message.get("kind") != "call_result" or message.get("call_id") != call_id:
                raise RuntimeError("backend call response did not match its request")
            return message.get("result")

    def count_tokens(self, text: str) -> int:
        return int(self._call("count_tokens", text=text))

    def count_tokens_many(self, texts: Sequence[str]) -> list[int]:
        return [int(value) for value in self._call("count_tokens_many", texts=list(texts))]

    def generate_many(self, requests: Iterable[GenerationRequest]) -> Iterator[BatchGenerationResult]:
        if self.connection is None:
            raise RuntimeError("backend process is not running")
        batch_id = uuid.uuid4().hex
        feeder_error: list[BaseException] = []
        window = threading.Condition()
        counters = {"submitted": 0, "completed": 0, "stopped": False}

        def feed() -> None:
            try:
                self.connection.send({"kind": "batch_start", "batch_id": batch_id})
                for index, request in enumerate(requests):
                    with window:
                        window.wait_for(
                            lambda: counters["stopped"]
                            or counters["submitted"] - counters["completed"] < self.config.batching.queue_capacity
                        )
                        if counters["stopped"]:
                            return
                    if request.request_id is None:
                        request = request.model_copy(update={"request_id": f"{batch_id}-{index:08d}"})
                    self.connection.send(
                        {"kind": "batch_item", "batch_id": batch_id, "request": request.model_dump(mode="json")}
                    )
                    with window:
                        counters["submitted"] += 1
                self.connection.send({"kind": "batch_end", "batch_id": batch_id})
            except BaseException as exc:
                feeder_error.append(exc)
                try:
                    self.connection.send({"kind": "batch_end", "batch_id": batch_id})
                except Exception:
                    pass

        with self._lock:
            feeder = threading.Thread(target=feed, name=f"backend-feeder-{batch_id[:8]}", daemon=True)
            feeder.start()
            try:
                while True:
                    message = self._get_result(self.config.batching.request_timeout_seconds)
                    self._raise_if_failed(message)
                    if message.get("batch_id") != batch_id:
                        raise RuntimeError("backend batch response did not match its request")
                    if message["kind"] == "batch_done":
                        break
                    if message["kind"] != "batch_result":
                        raise RuntimeError(f"unexpected backend batch message: {message['kind']}")
                    with window:
                        counters["completed"] += 1
                        window.notify_all()
                    yield BatchGenerationResult.model_validate(message["result"])
            finally:
                with window:
                    counters["stopped"] = True
                    window.notify_all()
                feeder.join(timeout=10)
            if feeder_error:
                raise RuntimeError(f"failed to submit backend batch: {feeder_error[0]}")

    def generate_json(self, request: GenerationRequest) -> BackendResponse:
        results = list(self.generate_many([request]))
        if len(results) != 1:
            raise RuntimeError("backend returned an invalid number of responses")
        result = results[0]
        if result.error is not None:
            raise RuntimeError(result.error)
        if result.response is None:
            raise RuntimeError("backend returned no response")
        return result.response

    def __exit__(self, *_args: Any) -> None:
        if self.connection is not None:
            try:
                self.connection.send({"kind": "close"})
            except (BrokenPipeError, EOFError):
                pass
        if self.process is not None:
            self.process.join(timeout=60)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=10)
        if self.connection is not None:
            self.connection.close()


register("backends", "fake", lambda config: FakeBackend(GenerationConfig.model_validate(config)))
register(
    "backends",
    "openai_compatible",
    lambda config: OpenAICompatibleBackend(GenerationConfig.model_validate(config)),
)
register("backends", "vllm_local", lambda config: VLLMLocalBackend(GenerationConfig.model_validate(config)))
