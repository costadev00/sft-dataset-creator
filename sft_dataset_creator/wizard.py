from __future__ import annotations

from pathlib import Path

from sft_dataset_creator.config import (
    CompositionConfig,
    CorpusSelection,
    DistributionConfig,
    GenerationConfig,
    ProjectConfig,
    SourceConfig,
    TargetConfig,
    gemma_cluster_preset,
    save_config,
)
from sft_dataset_creator.prompts import TASK_INSTRUCTIONS


def run_wizard(output: str | Path) -> ProjectConfig:
    try:
        from InquirerPy import inquirer
    except ImportError as exc:
        raise ImportError("the guided wizard requires InquirerPy") from exc
    name = inquirer.text(message="Project name:", default="synthetic-sft").execute()
    language = inquirer.text(message="Dataset language:", default="en").execute()
    source_kind = inquirer.select(
        message="Corpus source:",
        choices=[("Hugging Face Dataset", "huggingface"), ("Local JSON/JSONL/Parquet", "local")],
    ).execute()
    if source_kind == "huggingface":
        dataset = inquirer.text(message="Hugging Face dataset id:").execute()
        split = inquirer.text(message="Split:", default="train").execute()
        source = SourceConfig(
            plugin="huggingface",
            params={"dataset": dataset, "split": split, "streaming": True},
            field_map={"id": "id", "text": "text", "title": "title", "sections": "sections"},
        )
    else:
        path = inquirer.filepath(message="Local corpus path:", only_files=True).execute()
        source = SourceConfig(
            plugin="local",
            params={"path": path},
            field_map={"id": "id", "text": "text", "title": "title", "sections": "sections"},
        )
    selection_mode = inquirer.select(
        message="Corpus selection:", choices=[("Document count", "count"), ("Corpus fraction", "fraction")]
    ).execute()
    if selection_mode == "count":
        selection = CorpusSelection(count=int(inquirer.number(message="Documents to scan:", default=1000).execute()))
    else:
        percent = float(inquirer.number(message="Corpus percentage:", default=10.0).execute())
        selection = CorpusSelection(fraction=percent / 100.0)
    examples = int(inquirer.number(message="Final examples:", default=1000).execute())
    profile = inquirer.select(
        message="Document profile:",
        choices=[("Generic text corpus", None), ("Portuguese Wikipedia", "wikipedia_ptbr")],
        default=None,
    ).execute()
    tasks = inquirer.checkbox(
        message="Task recipes:",
        choices=list(TASK_INSTRUCTIONS),
        default=["closed_qa", "summarization", "information_extraction", "concept_explanation"],
        validate=lambda value: bool(value),
        invalid_message="Select at least one task.",
    ).execute()
    preset = inquirer.select(
        message="Model preset:",
        choices=[("4x RTX 4000 Ada: Gemma 26B + selective 31B", "gemma"), ("Fake backend for dry runs", "fake")],
    ).execute()
    if preset == "gemma":
        config = gemma_cluster_preset(
            name=name,
            source=source,
            selection=selection,
            examples=examples,
            language=language,
            profile=profile,
        )
    else:
        config = ProjectConfig(
            name=name,
            language=language,
            profile=profile,
            source=source,
            selection=selection,
            target=TargetConfig(examples=examples),
            generation=GenerationConfig(plugin="fake", model="fake-generator"),
        )
    weight = 1.0 / len(tasks)
    config = config.model_copy(
        update={
            "composition": CompositionConfig(
                tasks=DistributionConfig(weights={task: weight for task in tasks}),
                difficulties=DistributionConfig(weights={"easy": 0.25, "medium": 0.50, "hard": 0.25}),
            )
        }
    )
    save_config(config, output)
    return config
