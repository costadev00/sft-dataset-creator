from __future__ import annotations

import os
from pathlib import Path


def publish_output(
    output_dir: Path,
    *,
    repo_id: str,
    private: bool,
    token_env: str = "HF_TOKEN",
) -> str:
    token = os.getenv(token_env)
    if not token:
        raise EnvironmentError(f"missing Hugging Face token in {token_env}")
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError("publishing requires huggingface_hub") from exc
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=output_dir,
        commit_message="Publish reconstructed unchunked dataset",
    )
    return f"https://huggingface.co/datasets/{repo_id}"
