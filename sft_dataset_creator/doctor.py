from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

from sft_dataset_creator.backends import BackendProcess
from sft_dataset_creator.config import ProjectConfig
from sft_dataset_creator.models import ChatMessage, GenerationRequest
from sft_dataset_creator.prompts import DOCTOR_SYSTEM_PROMPT, DOCTOR_USER_PROMPT
from sft_dataset_creator.registry import available_plugins


SMOKE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ok"],
    "properties": {"ok": {"type": "boolean"}},
}


def _gpu_report() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=15)
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 5:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": int(parts[2]),
                    "memory_free_mb": int(parts[3]),
                    "driver": parts[4],
                }
            )
    return gpus


def _path_report(path: Path) -> dict[str, Any]:
    nearest = path
    while not nearest.exists() and nearest != nearest.parent:
        nearest = nearest.parent
    usage = shutil.disk_usage(nearest)
    return {
        "path": str(path),
        "nearest_existing": str(nearest),
        "exists": path.exists(),
        "writable": os.access(nearest, os.W_OK),
        "free_gb": round(usage.free / 1024**3, 2),
    }


def _smoke_backend(config) -> dict[str, Any]:
    request = GenerationRequest(
        slot_id="doctor",
        document_id="doctor",
        task="doctor",
        difficulty="easy",
        messages=[
            ChatMessage(role="system", content=DOCTOR_SYSTEM_PROMPT),
            ChatMessage(role="user", content=DOCTOR_USER_PROMPT),
        ],
        response_schema=SMOKE_SCHEMA,
        max_output_tokens=32,
    )
    with BackendProcess(config) as backend:
        response = backend.generate_json(request)
    return {"ok": response.payload.get("ok") is True, "model": response.model, "backend": response.backend}


def collect_doctor_report(config: ProjectConfig | None = None, *, smoke_models: bool = False) -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpus": _gpu_report(),
        "plugins": available_plugins(),
        "packages": {
            name: importlib.util.find_spec(name) is not None
            for name in ("pydantic", "typer", "rich", "datasets", "vllm", "openai", "pyarrow")
        },
        "paths": [],
        "models": [],
        "ready": True,
        "errors": [],
        "warnings": [],
    }
    if config is not None:
        report["paths"] = [
            _path_report(config.runtime.run_root),
            _path_report(config.runtime.cache_dir),
        ]
        for item in report["paths"]:
            if not item["writable"]:
                report["errors"].append(f"path is not writable: {item['nearest_existing']}")
        if config.source.plugin == "huggingface" and not report["packages"]["datasets"]:
            report["errors"].append("datasets is not installed; install the 'hf' extra")
        if config.source.plugin == "huggingface" and not config.source.params.get("revision"):
            report["warnings"].append(
                "Hugging Face dataset revision is not pinned; use --dataset-revision for reproducibility"
            )
        if "parquet" in config.output.containers and not report["packages"]["pyarrow"]:
            report["errors"].append("Parquet output requires the 'hf' extra with pyarrow")
        for stage, model_config in (
            ("generation", config.generation),
            ("evaluation", config.evaluation.llm),
        ):
            if model_config is None:
                continue
            if model_config.plugin == "vllm_local":
                if not report["packages"]["vllm"]:
                    report["errors"].append(f"{stage} requires the 'local' extra with vLLM")
                tensor_parallel_size = int(model_config.params.get("tensor_parallel_size", 1))
                if len(report["gpus"]) < tensor_parallel_size:
                    report["errors"].append(
                        f"fewer GPUs detected than {stage}.tensor_parallel_size"
                    )
            if model_config.plugin == "openai_compatible" and not report["packages"]["openai"]:
                report["errors"].append(f"{stage} requires the 'openai' extra")
        if config.evaluation.llm is None:
            report["warnings"].append(
                "LLM evaluation is disabled; deterministic gates cannot verify semantic grounding"
            )
        if smoke_models and not report["errors"]:
            for model_config in [config.generation, config.evaluation.llm]:
                if model_config is not None:
                    try:
                        smoke = _smoke_backend(model_config)
                        report["models"].append(smoke)
                        if not smoke["ok"]:
                            report["errors"].append(
                                f"model smoke test returned invalid output for {model_config.model}"
                            )
                    except Exception as exc:
                        report["errors"].append(f"model smoke test failed for {model_config.model}: {exc}")
    report["ready"] = not report["errors"]
    return report


def attach_environment(run_dir: str | Path, report: dict[str, Any]) -> None:
    path = Path(run_dir) / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["environment"] = {
        "python": report["python"],
        "platform": report["platform"],
        "gpus": report["gpus"],
        "plugins": report["plugins"],
        "packages": report["packages"],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
