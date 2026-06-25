from __future__ import annotations

import importlib.metadata
import json
import platform
import time
from itertools import cycle, islice
from pathlib import Path
from typing import Any, Literal

from sft_dataset_creator.backends import BackendProcess, create_token_counter
from sft_dataset_creator.config import BatchingConfig, GenerationConfig, ProjectConfig, save_config
from sft_dataset_creator.doctor import collect_doctor_report
from sft_dataset_creator.metrics import GPUMonitor, stage_metrics
from sft_dataset_creator.models import Document, GenerationRequest
from sft_dataset_creator.registry import create


PROFILE_CANDIDATES = (
    {"name": "async-8", "max_inflight": 8, "batched_tokens": 8_192, "memory": 0.90},
    {"name": "async-16", "max_inflight": 16, "batched_tokens": 16_384, "memory": 0.92},
    {"name": "async-32", "max_inflight": 32, "batched_tokens": 32_768, "memory": 0.92},
    {"name": "async-64", "max_inflight": 64, "batched_tokens": 65_536, "memory": 0.94},
)


def _sample_documents(config: ProjectConfig, samples: int) -> list[Document]:
    source = create("sources", config.source.plugin, config.source)
    documents = [document for document in islice(source.iter_documents(), samples) if document.text.strip()]
    if not documents:
        raise ValueError("tuning requires at least one non-empty source document")
    return documents


def _distribution_names(values: dict[str, Any] | None, fallback: str) -> list[str]:
    return sorted(values) if values else [fallback]


def _generation_requests(
    project: ProjectConfig,
    model_config: GenerationConfig,
    documents: list[Document],
    samples: int,
) -> list[GenerationRequest]:
    task_names = _distribution_names(
        project.composition.tasks.counts or project.composition.tasks.weights,
        "closed_qa",
    )
    difficulties = _distribution_names(
        project.composition.difficulties.counts or project.composition.difficulties.weights,
        "medium",
    )
    counter = create_token_counter(model_config)
    requests: list[GenerationRequest] = []
    for index, (document, task_name, difficulty) in enumerate(
        zip(cycle(documents), cycle(task_names), cycle(difficulties), strict=False)
    ):
        recipe = create("tasks", task_name, {"name": task_name, "language": project.language})
        request = recipe.build_request(
            document,
            slot_id=f"tune-{index:05d}",
            difficulty=difficulty,
            max_input_tokens=model_config.max_input_tokens,
            token_counter=counter.count_tokens,
        ).model_copy(
            update={
                "request_id": f"tune-generation-{index:05d}",
                "seed": index + 1,
                "max_output_tokens": model_config.max_output_tokens,
            }
        )
        requests.append(request)
        if len(requests) >= samples:
            break
    return requests


def _profile_config(base: GenerationConfig, profile: dict[str, Any] | None) -> GenerationConfig:
    if profile is None:
        return base.model_copy(
            update={"batching": base.batching.model_copy(update={"mode": "sequential", "max_inflight_requests": 1})}
        )
    inflight = int(profile["max_inflight"])
    params = dict(base.params)
    if base.plugin == "vllm_local":
        params.update(
            {
                "max_num_seqs": inflight,
                "max_num_batched_tokens": int(profile["batched_tokens"]),
                "gpu_memory_utilization": float(profile["memory"]),
                "enable_chunked_prefill": True,
                "enable_prefix_caching": True,
            }
        )
    batching = BatchingConfig(
        mode="async",
        max_inflight_requests=inflight,
        queue_capacity=inflight * 2,
        preparation_batch_size=max(64, inflight * 2),
        request_timeout_seconds=base.batching.request_timeout_seconds,
    )
    return base.model_copy(update={"params": params, "batching": batching})


def _benchmark(config: GenerationConfig, requests: list[GenerationRequest], name: str) -> dict[str, Any]:
    latencies: list[float] = []
    input_tokens = 0
    output_tokens = 0
    errors = 0
    max_queue_depth = 0
    monitor = GPUMonitor()
    startup = 0.0
    try:
        with BackendProcess(config) as backend:
            startup = backend.startup_seconds
            monitor.start()
            started = time.perf_counter()
            try:
                for result in backend.generate_many(requests):
                    latencies.append(result.latency_seconds)
                    max_queue_depth = max(max_queue_depth, result.queue_depth)
                    if result.error is not None or result.response is None:
                        errors += 1
                        continue
                    input_tokens += result.response.input_tokens or 0
                    output_tokens += result.response.output_tokens or 0
            finally:
                duration = time.perf_counter() - started
                gpu = monitor.stop()
        metrics = stage_metrics(
            requests=len(requests) - errors,
            duration_seconds=duration,
            startup_seconds=startup,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latencies=latencies,
            max_queue_depth=max_queue_depth,
            gpu=gpu,
        )
        return {"name": name, "stable": errors == 0, "errors": errors, "metrics": metrics}
    except Exception as exc:
        gpu = monitor.stop()
        return {
            "name": name,
            "stable": False,
            "errors": len(requests),
            "error": f"{type(exc).__name__}: {exc}",
            "metrics": {**stage_metrics(
                requests=0,
                duration_seconds=0.0,
                startup_seconds=startup,
                input_tokens=0,
                output_tokens=0,
                latencies=[],
                max_queue_depth=0,
                gpu=gpu,
            )},
        }


def _tune_stage(base: GenerationConfig, requests: list[GenerationRequest]) -> tuple[GenerationConfig, dict[str, Any]]:
    trials = [_benchmark(_profile_config(base, None), requests, "serial")]
    best_trial = trials[0] if trials[0]["stable"] else None
    best_config = _profile_config(base, None)
    stagnant = 0
    for profile in PROFILE_CANDIDATES:
        candidate_config = _profile_config(base, profile)
        trial = _benchmark(candidate_config, requests, str(profile["name"]))
        trials.append(trial)
        if not trial["stable"]:
            break
        throughput = float(trial["metrics"]["requests_per_minute"])
        best_throughput = (
            float(best_trial["metrics"]["requests_per_minute"])
            if best_trial is not None
            else 0.0
        )
        if throughput > best_throughput:
            improvement = (throughput - best_throughput) / best_throughput if best_throughput else 1.0
            best_trial = trial
            best_config = candidate_config
            stagnant = 0 if improvement > 0.03 else stagnant + 1
        else:
            stagnant += 1
        if stagnant >= 2:
            break
    if best_trial is None:
        raise RuntimeError("no stable inference profile was found")
    baseline = float(trials[0]["metrics"]["requests_per_minute"])
    selected = float(best_trial["metrics"]["requests_per_minute"])
    utilization = best_trial["metrics"]["gpu"].get("average_utilization_percent")
    return best_config, {
        "selected": best_trial["name"],
        "speedup": round(selected / baseline, 3) if baseline else None,
        "meets_2x_target": bool(baseline and selected >= baseline * 2),
        "meets_80_percent_gpu_target": utilization is not None and utilization >= 80.0,
        "selected_config": {
            "batching": best_config.batching.model_dump(mode="json"),
            "params": {
                key: best_config.params.get(key)
                for key in (
                    "gpu_memory_utilization",
                    "max_num_seqs",
                    "max_num_batched_tokens",
                    "enable_chunked_prefill",
                    "enable_prefix_caching",
                )
                if key in best_config.params
            },
        },
        "trials": trials,
    }


def tune_project(
    config: ProjectConfig,
    output: str | Path,
    *,
    stage: Literal["generation"] = "generation",
    samples: int = 32,
) -> tuple[ProjectConfig, Path]:
    if samples < 2:
        raise ValueError("tuning samples must be at least 2")
    if stage != "generation":
        raise ValueError("tuning only supports generation")
    documents = _sample_documents(config, samples)
    updated = config
    report: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "vllm": None,
        "samples": samples,
        "stages": {},
    }
    environment = collect_doctor_report()
    report["host"]["gpus"] = environment["gpus"]
    report["host"]["packages"] = environment["packages"]
    try:
        report["vllm"] = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        pass
    requests = _generation_requests(config, config.generation, documents, samples)
    generation, stage_report = _tune_stage(config.generation, requests)
    updated = updated.model_copy(update={"generation": generation})
    report["stages"]["generation"] = stage_report
    output_path = save_config(updated, output)
    report_path = output_path.with_name(f"{output_path.stem}.tuning-report.json")
    report["resolved_config_hash"] = updated.config_hash
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return updated, report_path
