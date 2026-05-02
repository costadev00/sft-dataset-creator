from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import orjson
import psutil


ROOT_PATH = Path("/")
DEFAULT_DISK_PATHS = (Path("/"), Path("/mnt/disco1"), Path("/mnt/disco2"))


def parse_bool(value: bool | str | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "sim", "s", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "nao", "não", "off"}:
        return False
    raise ValueError(f"Valor booleano inválido: {value!r}")


def gb_to_bytes(value: float) -> int:
    return int(value * 1024**3)


def bytes_to_gb(value: int | float) -> float:
    return float(value) / float(1024**3)


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path).expanduser()
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def nearest_existing_path(path: str | Path) -> Path:
    current = Path(path).expanduser()
    if current.exists():
        return current
    for parent in current.parents:
        if parent.exists():
            return parent
    return ROOT_PATH


def disk_usage_for(path: str | Path) -> shutil._ntuple_diskusage:
    return shutil.disk_usage(nearest_existing_path(path))


def free_gb_for(path: str | Path) -> float:
    return bytes_to_gb(disk_usage_for(path).free)


def path_on_root_device(path: str | Path) -> bool:
    try:
        root_dev = os.stat(ROOT_PATH).st_dev
        path_dev = os.stat(nearest_existing_path(path)).st_dev
        return root_dev == path_dev
    except OSError:
        return False


def safe_size(path: str | Path) -> int:
    path_obj = Path(path)
    if not path_obj.exists():
        return 0
    if path_obj.is_file():
        return path_obj.stat().st_size
    total = 0
    for child in path_obj.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def line_count(path: str | Path) -> int:
    path_obj = Path(path)
    if not path_obj.exists():
        return 0
    count = 0
    with path_obj.open("rb") as handle:
        for _ in handle:
            count += 1
    return count


def dumps_json(data: Any) -> str:
    return orjson.dumps(data, option=orjson.OPT_APPEND_NEWLINE | orjson.OPT_SORT_KEYS).decode("utf-8")


class JsonlWriter:
    def __init__(self, path: str | Path, append: bool = True, flush_every: int = 25) -> None:
        self.path = Path(path)
        self.append = append
        self.flush_every = max(1, flush_every)
        self._handle = None
        self._count_since_flush = 0

    def __enter__(self) -> "JsonlWriter":
        ensure_dir(self.path.parent)
        mode = "ab" if self.append else "wb"
        self._handle = self.path.open(mode)
        return self

    def write(self, data: Any) -> None:
        if self._handle is None:
            raise RuntimeError("JsonlWriter deve ser usado como context manager")
        self._handle.write(orjson.dumps(data, option=orjson.OPT_SORT_KEYS))
        self._handle.write(b"\n")
        self._count_since_flush += 1
        if self._count_since_flush >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._count_since_flush = 0

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._handle is not None:
            self.flush()
            self._handle.close()


def append_jsonl(path: str | Path, record: Any) -> None:
    ensure_dir(Path(path).parent)
    with Path(path).open("ab") as handle:
        handle.write(orjson.dumps(record, option=orjson.OPT_SORT_KEYS))
        handle.write(b"\n")


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    path_obj = Path(path)
    if not path_obj.exists():
        return
    with path_obj.open("rb") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield orjson.loads(line)


def load_jsonl_map(path: str | Path, key: str) -> dict[Any, dict[str, Any]]:
    output: dict[Any, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        if key in row:
            output[row[key]] = row
    return output


def strip_json_markdown(text: str) -> str:
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    return cleaned


def extract_json_text(text: str) -> str:
    cleaned = strip_json_markdown(text)
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first >= 0 and last > first:
        return cleaned[first : last + 1]
    raise ValueError("Não foi possível localizar um objeto JSON na resposta do modelo")


def parse_json_maybe_markdown(text: str) -> dict[str, Any]:
    return orjson.loads(extract_json_text(text))


def get_gpu_info() -> list[dict[str, Any]]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    gpus: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            memory_mb = int(parts[2])
        except ValueError:
            memory_mb = 0
        gpus.append({"index": parts[0], "name": parts[1], "memory_total_mb": memory_mb})
    return gpus


def get_ram_info() -> dict[str, float]:
    memory = psutil.virtual_memory()
    return {
        "total_gb": bytes_to_gb(memory.total),
        "available_gb": bytes_to_gb(memory.available),
    }


def get_disk_report(paths: Iterable[str | Path] = DEFAULT_DISK_PATHS) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for path in paths:
        path_obj = Path(path)
        exists = path_obj.exists()
        usage = disk_usage_for(path_obj)
        report[str(path_obj)] = {
            "exists": exists,
            "total_gb": bytes_to_gb(usage.total),
            "used_gb": bytes_to_gb(usage.used),
            "free_gb": bytes_to_gb(usage.free),
            "percent_used": (usage.used / usage.total * 100.0) if usage.total else 0.0,
            "on_root_device": path_on_root_device(path_obj),
        }
    return report


@dataclass(slots=True)
class StorageCheck:
    ready: bool
    warnings: list[str]
    errors: list[str]
    free_gb: dict[str, float]


def validate_storage_paths(
    output_dir: str | Path,
    cache_dir: str | Path,
    tmp_dir: str | Path,
    min_free_output_gb: float = 50.0,
    min_free_cache_gb: float = 100.0,
) -> StorageCheck:
    ensure_dir(output_dir)
    ensure_dir(cache_dir)
    ensure_dir(tmp_dir)

    warnings: list[str] = []
    errors: list[str] = []
    free = {
        "root": free_gb_for(ROOT_PATH),
        "output_dir": free_gb_for(output_dir),
        "cache_dir": free_gb_for(cache_dir),
        "tmp_dir": free_gb_for(tmp_dir),
    }

    for label, path in {
        "OUTPUT_DIR": output_dir,
        "CACHE_DIR": cache_dir,
        "TMPDIR": tmp_dir,
    }.items():
        if path_on_root_device(path):
            warnings.append(f"{label} aponta para o mesmo filesystem do disco raiz: {Path(path)}")

    if free["root"] < 100.0:
        warnings.append(f"Disco raiz com menos de 100 GB livres: {free['root']:.1f} GB")
    if free["output_dir"] < min_free_output_gb:
        errors.append(
            f"OUTPUT_DIR precisa de pelo menos {min_free_output_gb:.1f} GB livres; "
            f"encontrado {free['output_dir']:.1f} GB em {Path(output_dir)}"
        )
    if free["cache_dir"] < min_free_cache_gb:
        errors.append(
            f"CACHE_DIR precisa de pelo menos {min_free_cache_gb:.1f} GB livres; "
            f"encontrado {free['cache_dir']:.1f} GB em {Path(cache_dir)}"
        )

    return StorageCheck(ready=not errors, warnings=warnings, errors=errors, free_gb=free)


def truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip()


def normalize_page_id(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return abs(hash(str(value))) % 10_000_000_000


def record_id_for(page_id: Any) -> str:
    return f"wiki-ptbr-{normalize_page_id(page_id)}"

