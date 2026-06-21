from __future__ import annotations

import pytest
from pydantic import ValidationError

from sft_dataset_creator.config import CorpusSelection, ProjectConfig, load_config, save_config


def test_selection_requires_exactly_one_size() -> None:
    with pytest.raises(ValidationError):
        CorpusSelection()
    with pytest.raises(ValidationError):
        CorpusSelection(count=10, fraction=0.5)


def test_config_round_trip_and_hash(project_config, tmp_path) -> None:
    path = save_config(project_config, tmp_path / "project.json")
    loaded = load_config(path)
    assert loaded == project_config
    assert loaded.config_hash == project_config.config_hash


def test_config_rejects_unknown_keys(project_config) -> None:
    payload = project_config.model_dump(mode="json")
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(payload)
