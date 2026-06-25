from __future__ import annotations

import hashlib
import math
import platform
import threading
import time
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from sft_dataset_creator.backends import BackendProcess, create_token_counter
from sft_dataset_creator.chunking import select_chunk
from sft_dataset_creator.config import GenerationConfig, ProjectConfig
from sft_dataset_creator.exporters import export_run
from sft_dataset_creator.metrics import GPUMonitor, stage_metrics
from sft_dataset_creator.models import (
    DatasetPlan,
    GenerationRequest,
    PlannedSlot,
    RunManifest,
    RunReport,
    SFTCandidate,
    utc_now,
)
from sft_dataset_creator.planner import load_document_snapshot
from sft_dataset_creator.registry import create
from sft_dataset_creator.state import RunState


EventCallback = Callable[[str, dict], None]


@dataclass(frozen=True)
class GenerationJob:
    slot: PlannedSlot
    attempt: int
    document_id: str

    @property
    def request_id(self) -> str:
        return f"{self.slot.id}-a{self.attempt}"


def _emit(callback: EventCallback | None, kind: str, payload: dict) -> None:
    if callback:
        callback(kind, payload)


def _update_manifest(run_dir: Path, status: str) -> None:
    path = run_dir / "manifest.json"
    manifest = RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
    manifest = manifest.model_copy(update={"status": status, "updated_at": utc_now()})
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def _request_seed(seed: int, request_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{request_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _document_for_attempt(
    plan: DatasetPlan,
    slot_id: str,
    original_id: str,
    attempt: int,
    same_attempts: int,
    document_loads: dict[str, int],
    maximum: int,
) -> str:
    if attempt <= same_attempts or not plan.reserve_document_ids:
        return original_id
    digest = hashlib.sha256(f"{plan.seed}:{slot_id}:{attempt}".encode("utf-8")).digest()
    start = int.from_bytes(digest[:8], "big") % len(plan.reserve_document_ids)
    for offset in range(len(plan.reserve_document_ids)):
        candidate = plan.reserve_document_ids[(start + offset) % len(plan.reserve_document_ids)]
        if document_loads.get(candidate, 0) < maximum:
            return candidate
    return original_id


def _generation_jobs(state: RunState, plan: DatasetPlan, config: ProjectConfig) -> list[GenerationJob]:
    maximum_attempts = math.ceil(config.target.examples * config.target.max_total_attempt_multiplier)
    remaining = maximum_attempts - state.counts()["attempted"]
    if remaining <= 0:
        return []
    existing = state.attempt_keys()
    document_loads = state.accepted_document_counts()
    jobs: list[GenerationJob] = []
    for slot, attempt in state.pending_slots(config.target.max_attempts_per_slot):
        if (slot.id, attempt) in existing:
            continue
        document_id = _document_for_attempt(
            plan,
            slot.id,
            slot.document_id,
            attempt,
            config.target.same_document_attempts,
            document_loads,
            config.target.per_document.maximum,
        )
        jobs.append(GenerationJob(slot=slot, attempt=attempt, document_id=document_id))
        if attempt > config.target.same_document_attempts:
            document_loads[document_id] = document_loads.get(document_id, 0) + 1
        if len(jobs) >= remaining:
            break
    return jobs


def _prepare_request_chunk(
    jobs: Sequence[GenerationJob],
    documents: dict[str, Any],
    config: GenerationConfig,
    language: str,
    plan_seed: int,
    counter: Any,
) -> list[GenerationRequest]:
    prepared: list[GenerationRequest | None] = [None] * len(jobs)
    groups: dict[str, list[tuple[int, GenerationJob]]] = defaultdict(list)
    for index, job in enumerate(jobs):
        groups[job.slot.task].append((index, job))
    for task_name, entries in groups.items():
        recipe = create("tasks", task_name, {"name": task_name, "language": language})
        generation_entries = [
            (
                index,
                job,
                select_chunk(
                    documents[job.document_id],
                    job.slot.chunk_id,
                    fallback_index=job.slot.ordinal + job.attempt - 2,
                ),
            )
            for index, job in entries
        ]
        batch_builder = getattr(recipe, "build_requests", None)
        if batch_builder is not None:
            requests = batch_builder(
                [
                    (document, job.slot.id, job.slot.difficulty)
                    for _index, job, document in generation_entries
                ],
                max_input_tokens=config.max_input_tokens,
                token_counter_many=counter.count_tokens_many,
            )
        else:
            requests = [
                recipe.build_request(
                    document,
                    slot_id=job.slot.id,
                    difficulty=job.slot.difficulty,
                    max_input_tokens=config.max_input_tokens,
                    token_counter=counter.count_tokens,
                )
                for _index, job, document in generation_entries
            ]
        for (index, job, document), request in zip(generation_entries, requests, strict=True):
            selected_chunk_id = document.metadata.get("selected_chunk_id")
            prepared[index] = request.model_copy(
                update={
                    "request_id": job.request_id,
                    "seed": _request_seed(plan_seed, job.request_id),
                    "max_output_tokens": config.max_output_tokens,
                    "metadata": {**request.metadata, "chunk_id": selected_chunk_id},
                }
            )
    return [request for request in prepared if request is not None]


def _request_stream(
    jobs: Sequence[GenerationJob],
    documents: dict[str, Any],
    config: GenerationConfig,
    language: str,
    plan_seed: int,
    counter: Any,
    prepared: dict[str, GenerationRequest],
    prepared_lock: threading.Lock,
) -> Iterator[GenerationRequest]:
    size = config.batching.preparation_batch_size
    for start in range(0, len(jobs), size):
        requests = _prepare_request_chunk(
            jobs[start : start + size], documents, config, language, plan_seed, counter
        )
        for request in requests:
            with prepared_lock:
                prepared[str(request.request_id)] = request
            yield request


def _empty_stage_metrics() -> dict[str, Any]:
    return stage_metrics(
        requests=0,
        duration_seconds=0.0,
        startup_seconds=0.0,
        input_tokens=0,
        output_tokens=0,
        latencies=[],
        max_queue_depth=0,
        gpu={"sample_count": 0, "gpu_count": 0, "average_utilization_percent": None, "peak_memory_used_mb": None},
    )


def _aggregate_stage_metrics(rounds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rounds:
        return {**_empty_stage_metrics(), "rounds": []}
    requests = sum(int(item["requests"]) for item in rounds)
    duration = sum(float(item["duration_seconds"]) for item in rounds)
    startup = sum(float(item["model_startup_seconds"]) for item in rounds)
    input_tokens = sum(int(item["input_tokens"]) for item in rounds)
    output_tokens = sum(int(item["output_tokens"]) for item in rounds)
    weighted_p50 = (
        sum(float(item["latency_p50_seconds"]) * int(item["requests"]) for item in rounds) / requests
        if requests
        else 0.0
    )
    gpu_samples = sum(int(item["gpu"].get("sample_count") or 0) for item in rounds)
    utilization_weight = sum(
        float(item["gpu"].get("average_utilization_percent") or 0.0)
        * int(item["gpu"].get("sample_count") or 0)
        for item in rounds
    )
    peak_values = [
        float(item["gpu"]["peak_memory_used_mb"])
        for item in rounds
        if item["gpu"].get("peak_memory_used_mb") is not None
    ]
    return {
        "requests": requests,
        "duration_seconds": round(duration, 3),
        "model_startup_seconds": round(startup, 3),
        "requests_per_minute": round(requests * 60 / duration, 3) if duration else 0.0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_second": round((input_tokens + output_tokens) / duration, 3) if duration else 0.0,
        "latency_p50_seconds": round(weighted_p50, 3),
        "latency_p95_seconds": max(float(item["latency_p95_seconds"]) for item in rounds),
        "max_worker_queue_depth": max(int(item["max_worker_queue_depth"]) for item in rounds),
        "gpu": {
            "sample_count": gpu_samples,
            "gpu_count": max(int(item["gpu"].get("gpu_count") or 0) for item in rounds),
            "average_utilization_percent": (
                round(utilization_weight / gpu_samples, 2) if gpu_samples else None
            ),
            "peak_memory_used_mb": max(peak_values) if peak_values else None,
        },
        "rounds": list(rounds),
    }


def _generate(
    state: RunState,
    jobs: Sequence[GenerationJob],
    documents: dict[str, Any],
    plan: DatasetPlan,
    config: ProjectConfig,
    callback: EventCallback | None,
    round_number: int,
    counter: Any,
    generator_process: BackendProcess | None = None,
    record_startup: bool = True,
) -> dict[str, Any]:
    if not jobs:
        return _empty_stage_metrics()
    _emit(callback, "generation_started", {"round": round_number, "slots": len(jobs)})
    job_by_id = {job.request_id: job for job in jobs}
    prepared: dict[str, GenerationRequest] = {}
    prepared_lock = threading.Lock()
    latencies: list[float] = []
    input_tokens = 0
    output_tokens = 0
    max_queue_depth = 0
    generated = 0
    monitor = GPUMonitor()
    started = 0.0
    startup_seconds = 0.0
    duration = 0.0
    gpu: dict[str, Any] = {}
    backend_context = (
        nullcontext(generator_process)
        if generator_process is not None
        else BackendProcess(config.generation)
    )
    with backend_context as generator:
        startup_seconds = generator.startup_seconds if record_startup else 0.0
        monitor.start()
        started = time.perf_counter()
        stream = _request_stream(
            jobs,
            documents,
            config.generation,
            config.language,
            plan.seed,
            counter,
            prepared,
            prepared_lock,
        )
        try:
            for result in generator.generate_many(stream):
                latencies.append(result.latency_seconds)
                max_queue_depth = max(max_queue_depth, result.queue_depth)
                job = job_by_id[result.request_id]
                with prepared_lock:
                    request = prepared.pop(result.request_id, None)
                document = documents[job.document_id]
                if result.error is not None or result.response is None:
                    state.record_attempt(
                        None,
                        slot_id=job.slot.id,
                        attempt=job.attempt,
                        document_id=job.document_id,
                        request_json=request.model_dump_json() if request and config.runtime.store_model_io else None,
                        error=result.error or "backend returned no response",
                        request_id=result.request_id,
                        latency_seconds=result.latency_seconds,
                    )
                    continue
                response = result.response
                try:
                    recipe = create("tasks", job.slot.task, {"name": job.slot.task})
                    candidate = recipe.candidate_from_response(
                        response,
                        document=document,
                        slot_id=job.slot.id,
                        attempt=job.attempt,
                        generator=config.generation.plugin,
                        model=config.generation.model,
                    )
                    candidate = candidate.model_copy(
                        update={
                            "difficulty": job.slot.difficulty,
                            "metadata": {**candidate.metadata, **(request.metadata if request else {})},
                        }
                    )
                    state.record_attempt(
                        candidate,
                        slot_id=job.slot.id,
                        attempt=job.attempt,
                        document_id=job.document_id,
                        request_json=request.model_dump_json() if request and config.runtime.store_model_io else None,
                        response_json=response.model_dump_json() if config.runtime.store_model_io else None,
                        request_id=result.request_id,
                        input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens,
                        latency_seconds=result.latency_seconds,
                    )
                    generated += 1
                except Exception as exc:
                    state.record_attempt(
                        None,
                        slot_id=job.slot.id,
                        attempt=job.attempt,
                        document_id=job.document_id,
                        request_json=request.model_dump_json() if request and config.runtime.store_model_io else None,
                        response_json=response.model_dump_json() if config.runtime.store_model_io else None,
                        error=str(exc),
                        request_id=result.request_id,
                        input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens,
                        latency_seconds=result.latency_seconds,
                    )
                input_tokens += response.input_tokens or 0
                output_tokens += response.output_tokens or 0
        finally:
            duration = time.perf_counter() - started
            gpu = monitor.stop()
    _emit(callback, "generation_finished", {"round": round_number, "generated": generated})
    return {
        **stage_metrics(
            requests=len(jobs),
            duration_seconds=duration,
            startup_seconds=startup_seconds,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latencies=latencies,
            max_queue_depth=max_queue_depth,
            gpu=gpu,
        ),
        "round": round_number,
    }


def _evaluate_generated(
    state: RunState,
    documents: dict[str, Any],
    plan: DatasetPlan,
    config: ProjectConfig,
    callback: EventCallback | None,
    round_number: int,
) -> dict[str, Any]:
    strategy = create("evaluators", config.evaluation.plugin, config.evaluation)
    ordinal = {slot.id: slot.ordinal for slot in plan.slots}
    candidates = sorted(state.generated_candidates(), key=lambda value: (ordinal[value.slot_id], value.attempt))
    if not candidates:
        return _empty_stage_metrics()
    accepted = list(state.accepted_candidates())
    accepted_slots = {candidate.slot_id for candidate in accepted}
    for candidate in candidates:
        if candidate.slot_id in accepted_slots:
            state.mark_superseded(candidate)
            continue
        document = documents[candidate.document_id]
        evaluation = strategy.deterministic(candidate, document, accepted)
        state.record_evaluation(candidate, evaluation)
        if evaluation.verdict == "accept":
            accepted.append(candidate)
            accepted_slots.add(candidate.slot_id)
    return {**_empty_stage_metrics(), "round": round_number}


def execute_plan(
    plan: DatasetPlan,
    config: ProjectConfig,
    *,
    run_dir: str | Path | None = None,
    callback: EventCallback | None = None,
    auto_export: bool = True,
) -> RunReport:
    if plan.config_hash != config.config_hash:
        raise ValueError("plan and resolved configuration hashes do not match")
    target_dir = Path(run_dir) if run_dir else Path(plan.corpus_snapshot).parent
    documents = load_document_snapshot(plan.corpus_snapshot)
    database = target_dir / "run.db"
    _update_manifest(target_dir, "running")
    with RunState(database) as state:
        state.initialize(plan)
        generation_rounds: list[dict[str, Any]] = []
        evaluation_rounds: list[dict[str, Any]] = []
        recovered_metrics = _evaluate_generated(state, documents, plan, config, callback, 0)
        if recovered_metrics["requests"]:
            evaluation_rounds.append(recovered_metrics)
        round_number = 0
        counts = state.counts()
        jobs = _generation_jobs(state, plan, config)
        counter = create_token_counter(config.generation) if jobs else None

        def execute_round(generator_process: BackendProcess | None = None) -> None:
            nonlocal counts, jobs, round_number
            round_number += 1
            generation_rounds.append(
                _generate(
                    state,
                    jobs,
                    documents,
                    plan,
                    config,
                    callback,
                    round_number,
                    counter,
                    generator_process=generator_process,
                    record_startup=generator_process is None or round_number == 1,
                )
            )
            evaluation_rounds.append(
                _evaluate_generated(state, documents, plan, config, callback, round_number)
            )
            counts = state.counts()
            _emit(callback, "round_finished", {"round": round_number, **counts})
            jobs = _generation_jobs(state, plan, config)

        if jobs:
            with BackendProcess(config.generation) as generator_process:
                while jobs and counts["accepted"] < config.target.examples:
                    execute_round(generator_process)

        generation_metrics = _aggregate_stage_metrics(generation_rounds)
        evaluation_metrics = _aggregate_stage_metrics(evaluation_rounds)
        counts = state.counts()
        status = "completed" if counts["accepted"] >= config.target.examples else "partial"
        tokens = state.token_totals()
        state.event("run_finished", {"status": status, **counts})
        report = RunReport(
            run_id=plan.run_id,
            status=status,
            target_examples=counts["target"],
            accepted_examples=counts["accepted"],
            attempted_examples=counts["attempted"],
            rejected_examples=counts["rejected"],
            reviewed_examples=counts["reviewed"],
            speculative_examples=counts["speculative"],
            llm_judged_examples=0,
            deficits=state.deficits(),
            metrics={
                "acceptance_rate": counts["accepted"] / counts["attempted"] if counts["attempted"] else 0.0,
                "llm_judge_coverage": 0.0,
                "semantic_judge_configured": False,
                "python": platform.python_version(),
                "generation": generation_metrics,
                "evaluation": evaluation_metrics,
                "tokens": tokens,
            },
        )
    (target_dir / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    _update_manifest(target_dir, status)
    if auto_export:
        export_run(target_dir, config=config)
    _emit(callback, "run_finished", report.model_dump(mode="json"))
    return report
