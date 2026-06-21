from __future__ import annotations

from sft_dataset_creator.config import save_config
from sft_dataset_creator.publisher import publish_run


def test_publish_run_uses_generic_huggingface_repo(project_config, tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "exports").mkdir(parents=True)
    (run_dir / "exports" / "README.md").write_text("# Dataset\n", encoding="utf-8")
    save_config(project_config, run_dir / "config.resolved.json")
    calls = []

    class FakeApi:
        def __init__(self, token):
            calls.append(("init", token))

        def create_repo(self, repo_id, **kwargs):
            calls.append(("create", repo_id, kwargs))

        def upload_folder(self, **kwargs):
            calls.append(("upload", kwargs))

    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    url = publish_run(run_dir, repo_id="owner/synthetic", private=True)
    assert url.endswith("owner/synthetic")
    assert calls[1][2]["private"] is True
