from wiki_if_builder.cli import collect_doctor_report
from wiki_if_builder.config import load_config
from wiki_if_builder.llm_client import RoundRobinLLMClient


def test_doctor_identifies_disk_paths(tmp_path):
    config = load_config(
        work_dir=tmp_path / "work",
        cache_dir=tmp_path / "work" / "cache",
        output_dir=tmp_path / "work" / "outputs",
        hf_home=tmp_path / "work" / "cache" / "huggingface",
        hf_datasets_cache=tmp_path / "work" / "cache" / "huggingface" / "datasets",
        transformers_cache=tmp_path / "work" / "cache" / "huggingface" / "transformers",
        model_cache_dir=tmp_path / "work" / "models",
        tmp_dir=tmp_path / "tmp",
        min_free_output_gb=0,
        min_free_cache_gb=0,
    )
    report = collect_doctor_report(config)
    assert "/" in report["disks"]
    assert "/mnt/disco1" in report["disks"]
    assert "/mnt/disco2" in report["disks"]
    assert "free_gb" in report["disks"]["/"]


def test_llm_client_round_robin_base_urls():
    client = RoundRobinLLMClient(
        base_urls=["http://localhost:8000/v1", "http://localhost:8001/v1"],
        api_key="local-token",
        model="gemma-local",
    )
    assert client.next_base_url() == "http://localhost:8000/v1"
    assert client.next_base_url() == "http://localhost:8001/v1"
    assert client.next_base_url() == "http://localhost:8000/v1"

