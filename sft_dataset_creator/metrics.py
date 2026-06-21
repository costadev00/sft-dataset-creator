from __future__ import annotations

import statistics
import subprocess
import threading
from collections.abc import Sequence
from typing import Any


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


class GPUMonitor:
    def __init__(self, interval_seconds: float = 1.0) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[list[tuple[float, float]]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="gpu-monitor", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=5)
                sample: list[tuple[float, float]] = []
                for line in output.splitlines():
                    utilization, memory = [float(part.strip()) for part in line.split(",")[:2]]
                    sample.append((utilization, memory))
                if sample:
                    self.samples.append(sample)
            except (FileNotFoundError, subprocess.SubprocessError, ValueError):
                return
            self._stop.wait(self.interval_seconds)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 2))
        utilizations = [value for sample in self.samples for value, _memory in sample]
        memories = [value for sample in self.samples for _utilization, value in sample]
        return {
            "sample_count": len(self.samples),
            "gpu_count": max((len(sample) for sample in self.samples), default=0),
            "average_utilization_percent": round(statistics.fmean(utilizations), 2) if utilizations else None,
            "peak_memory_used_mb": round(max(memories), 2) if memories else None,
        }


def stage_metrics(
    *,
    requests: int,
    duration_seconds: float,
    startup_seconds: float,
    input_tokens: int,
    output_tokens: int,
    latencies: Sequence[float],
    max_queue_depth: int,
    gpu: dict[str, Any],
) -> dict[str, Any]:
    return {
        "requests": requests,
        "duration_seconds": round(duration_seconds, 3),
        "model_startup_seconds": round(startup_seconds, 3),
        "requests_per_minute": round(requests * 60 / duration_seconds, 3) if duration_seconds else 0.0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_second": round((input_tokens + output_tokens) / duration_seconds, 3) if duration_seconds else 0.0,
        "latency_p50_seconds": round(percentile(latencies, 0.50), 3),
        "latency_p95_seconds": round(percentile(latencies, 0.95), 3),
        "max_worker_queue_depth": max_queue_depth,
        "gpu": gpu,
    }
