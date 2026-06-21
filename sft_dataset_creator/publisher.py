from __future__ import annotations

import os
from pathlib import Path

from sft_dataset_creator.config import load_config


def publish_run(
    run_dir: str | Path,
    *,
    repo_id: str | None = None,
    private: bool | None = None,
) -> str:
    root = Path(run_dir)
    config = load_config(root / "config.resolved.json")
    target_repo = repo_id or config.publish.repo_id
    if not target_repo:
        raise ValueError("publish requires a repo_id argument or publish.repo_id in config")
    token = os.getenv(config.publish.token_env)
    if not token:
        raise EnvironmentError(f"missing Hugging Face token in {config.publish.token_env}")
    exports = root / "exports"
    if not exports.exists():
        raise FileNotFoundError(f"missing exports directory: {exports}")
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError("publishing requires the 'hf' extra") from exc
    visibility = config.publish.private if private is None else private
    api = HfApi(token=token)
    api.create_repo(target_repo, repo_type="dataset", private=visibility, exist_ok=True)
    api.upload_folder(
        repo_id=target_repo,
        repo_type="dataset",
        folder_path=exports,
        commit_message=f"Publish synthetic SFT run {root.name}",
        delete_patterns=[
            "dataset_info.json",
            "generation_info.json",
            "messages/*",
            "prompt_completion/*",
            "alpaca/*",
        ],
    )
    return f"https://huggingface.co/datasets/{target_repo}"
