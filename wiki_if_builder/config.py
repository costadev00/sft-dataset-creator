from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator

from wiki_if_builder import __version__
from wiki_if_builder.utils import ensure_dir, parse_bool, validate_storage_paths


DEFAULT_WORK_DIR = Path("/mnt/disco1/wiki-if-builder")
DEFAULT_CACHE_DIR = DEFAULT_WORK_DIR / "cache"
DEFAULT_OUTPUT_DIR = DEFAULT_WORK_DIR / "outputs"
DEFAULT_HF_HOME = DEFAULT_CACHE_DIR / "huggingface"
DEFAULT_HF_DATASETS_CACHE = DEFAULT_HF_HOME / "datasets"
DEFAULT_TRANSFORMERS_CACHE = DEFAULT_HF_HOME / "transformers"
DEFAULT_MODEL_CACHE_DIR = DEFAULT_WORK_DIR / "models"
DEFAULT_TMPDIR = Path("/mnt/disco2/wiki-if-builder/tmp")
DEFAULT_DATASET_NAME = "costadev00/wikipedia-pt-br-extract"
DEFAULT_LICENSE = "cc-by-sa-3.0"


class AppConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_name: str = DEFAULT_DATASET_NAME
    split: str = "train"
    work_dir: Path = DEFAULT_WORK_DIR
    cache_dir: Path = DEFAULT_CACHE_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    hf_home: Path = DEFAULT_HF_HOME
    hf_datasets_cache: Path = DEFAULT_HF_DATASETS_CACHE
    transformers_cache: Path = DEFAULT_TRANSFORMERS_CACHE
    model_cache_dir: Path = DEFAULT_MODEL_CACHE_DIR
    tmp_dir: Path = DEFAULT_TMPDIR

    openai_base_url: str = "http://localhost:8000/v1"
    openai_base_urls: list[str] = Field(default_factory=list)
    openai_api_key: str = "local-token"
    model_name: str = "gemma-local"
    judge_model_name: str = "gemma-local"
    llm_timeout_seconds: float = 180.0

    hf_token: str | None = None
    hf_labels_repo_id: str = "costadev00/wikipedia-pt-br-article-labels-gemma"
    hf_if_repo_id: str = "costadev00/wikipedia-pt-br-instructions-gemma"
    hf_private: bool = True

    pipeline_version: str = __version__
    max_articles: int | None = None
    max_input_chars: int = 120_000
    max_output_tokens: int = 4096
    candidates_per_article: int = 3
    enable_judge: bool = False
    dry_run: bool = False
    resume: bool = False
    include_review: bool = False
    private: bool = True
    num_workers: int = 1
    max_concurrent_llm_calls: int | None = None
    min_free_output_gb: float = 50.0
    min_free_cache_gb: float = 100.0
    log_every: int = 25

    @field_validator(
        "work_dir",
        "cache_dir",
        "output_dir",
        "hf_home",
        "hf_datasets_cache",
        "transformers_cache",
        "model_cache_dir",
        "tmp_dir",
        mode="before",
    )
    @classmethod
    def _expand_path(cls, value: Any) -> Path:
        return Path(value).expanduser()

    @field_validator("openai_base_urls", mode="before")
    @classmethod
    def _parse_base_urls(cls, value: Any) -> list[str]:
        if value is None or value == "":
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return list(value)

    @property
    def effective_base_urls(self) -> list[str]:
        return self.openai_base_urls or [self.openai_base_url]

    @property
    def resolved_max_concurrent_llm_calls(self) -> int:
        if self.max_concurrent_llm_calls is not None:
            return max(1, int(self.max_concurrent_llm_calls))
        return max(1, len(self.effective_base_urls))

    @property
    def intermediate_dir(self) -> Path:
        return self.output_dir / "intermediate"

    @property
    def document_labels_dir(self) -> Path:
        return self.output_dir / "document_labels"

    @property
    def instruction_following_dir(self) -> Path:
        return self.output_dir / "instruction_following"

    def ensure_directories(self) -> None:
        for path in (
            self.work_dir,
            self.cache_dir,
            self.output_dir,
            self.hf_home,
            self.hf_datasets_cache,
            self.transformers_cache,
            self.model_cache_dir,
            self.tmp_dir,
            self.intermediate_dir,
        ):
            ensure_dir(path)

    def apply_environment(self) -> None:
        os.environ["WORK_DIR"] = str(self.work_dir)
        os.environ["CACHE_DIR"] = str(self.cache_dir)
        os.environ["OUTPUT_DIR"] = str(self.output_dir)
        os.environ["HF_HOME"] = str(self.hf_home)
        os.environ["HF_DATASETS_CACHE"] = str(self.hf_datasets_cache)
        os.environ["TRANSFORMERS_CACHE"] = str(self.transformers_cache)
        os.environ["MODEL_CACHE_DIR"] = str(self.model_cache_dir)
        os.environ["TMPDIR"] = str(self.tmp_dir)

    def validate_storage(self):
        self.ensure_directories()
        return validate_storage_paths(
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
            tmp_dir=self.tmp_dir,
            min_free_output_gb=self.min_free_output_gb,
            min_free_cache_gb=self.min_free_cache_gb,
        )


def _env(name: str, default: Any = None) -> Any:
    return os.getenv(name, default)


def _env_bool(name: str, default: bool) -> bool:
    return parse_bool(os.getenv(name), default=default)


def load_config(**overrides: Any) -> AppConfig:
    load_dotenv()
    values: dict[str, Any] = {
        "dataset_name": _env("HF_DATASET_NAME", DEFAULT_DATASET_NAME),
        "work_dir": _env("WORK_DIR", DEFAULT_WORK_DIR),
        "cache_dir": _env("CACHE_DIR", DEFAULT_CACHE_DIR),
        "output_dir": _env("OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        "hf_home": _env("HF_HOME", DEFAULT_HF_HOME),
        "hf_datasets_cache": _env("HF_DATASETS_CACHE", DEFAULT_HF_DATASETS_CACHE),
        "transformers_cache": _env("TRANSFORMERS_CACHE", DEFAULT_TRANSFORMERS_CACHE),
        "model_cache_dir": _env("MODEL_CACHE_DIR", DEFAULT_MODEL_CACHE_DIR),
        "tmp_dir": _env("TMPDIR", DEFAULT_TMPDIR),
        "openai_base_url": _env("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        "openai_base_urls": _env("OPENAI_BASE_URLS", ""),
        "openai_api_key": _env("OPENAI_API_KEY", "local-token"),
        "model_name": _env("MODEL_NAME", "gemma-local"),
        "judge_model_name": _env("JUDGE_MODEL_NAME", _env("MODEL_NAME", "gemma-local")),
        "hf_token": _env("HF_TOKEN", None),
        "hf_labels_repo_id": _env(
            "HF_LABELS_REPO_ID", "costadev00/wikipedia-pt-br-article-labels-gemma"
        ),
        "hf_if_repo_id": _env("HF_IF_REPO_ID", "costadev00/wikipedia-pt-br-instructions-gemma"),
        "hf_private": _env_bool("HF_PRIVATE", True),
        "pipeline_version": _env("PIPELINE_VERSION", __version__),
    }
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    config = AppConfig(**values)
    config.ensure_directories()
    config.apply_environment()
    return config

