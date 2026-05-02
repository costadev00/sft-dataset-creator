import pytest

from wiki_if_builder.hf_publisher import validate_dataset_publish_dir


def test_hf_publisher_validates_missing_readme(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "data.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="README.md"):
        validate_dataset_publish_dir(dataset_dir)


def test_hf_publisher_validates_missing_data_file(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "README.md").write_text("# Dataset\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="data.jsonl"):
        validate_dataset_publish_dir(dataset_dir)

