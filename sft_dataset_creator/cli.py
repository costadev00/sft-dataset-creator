from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from sft_dataset_creator.audit import create_audit_sample, score_audit
from sft_dataset_creator.config import (
    CompositionConfig,
    CorpusSelection,
    DistributionConfig,
    EvaluationConfig,
    GenerationConfig,
    OutputConfig,
    ProjectConfig,
    RuntimeConfig,
    SourceConfig,
    SplitConfig,
    TargetConfig,
    load_config,
    write_config_schema,
)
from sft_dataset_creator.doctor import attach_environment, collect_doctor_report
from sft_dataset_creator.engine import execute_plan
from sft_dataset_creator.exporters import export_run
from sft_dataset_creator.planner import build_plan, load_plan
from sft_dataset_creator.publisher import publish_run
from sft_dataset_creator.prompts import TASK_INSTRUCTIONS
from sft_dataset_creator.registry import available_plugins
from sft_dataset_creator.state import RunState
from sft_dataset_creator.tuning import tune_project


app = typer.Typer(
    name="sft-dataset",
    help="Plan, generate, evaluate, and publish reproducible synthetic SFT datasets.",
    no_args_is_help=True,
)
console = Console()


DEFAULT_MODEL = "google/gemma-4-26B-A4B-it"
DEFAULT_TASKS = {
    "closed_qa": 0.25,
    "summarization": 0.20,
    "information_extraction": 0.20,
    "concept_explanation": 0.20,
    "classification": 0.15,
}
DEFAULT_DIFFICULTIES = {"easy": 0.25, "medium": 0.50, "hard": 0.25}


def _key_values(values: list[str], option: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for value in values:
        key, separator, raw = value.partition("=")
        if not separator or not key.strip() or not raw.strip():
            raise typer.BadParameter(f"{option} values must use KEY=VALUE")
        try:
            parsed[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            parsed[key.strip()] = raw
    return parsed


def _weights(values: list[str], defaults: dict[str, float], option: str) -> dict[str, float]:
    if not values:
        return defaults
    parsed: dict[str, float] = {}
    for value in values:
        name, separator, raw = value.partition("=")
        if not name.strip():
            raise typer.BadParameter(f"{option} requires a non-empty name")
        try:
            parsed[name.strip()] = float(raw) if separator else 1.0
        except ValueError as exc:
            raise typer.BadParameter(f"{option} weights must be numeric: {value}") from exc
    return parsed


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _project_name(dataset: str) -> str:
    value = Path(dataset).stem if Path(dataset).suffix else dataset.rstrip("/").rsplit("/", 1)[-1]
    normalized = "".join(character.lower() if character.isalnum() else "-" for character in value)
    return normalized.strip("-") or "synthetic-sft"


def _direct_config(
    *,
    dataset: str,
    examples: int,
    name: str | None,
    language: str,
    source_plugin: str,
    subset: str | None,
    split: str,
    streaming: bool,
    source_format: str | None,
    cache_dir: Path,
    id_field: str,
    text_field: str,
    title_field: str,
    sections_field: str,
    license_field: str,
    document_count: int | None,
    selection_fraction: float | None,
    seed: int,
    profile: str | None,
    per_document_minimum: int,
    per_document_maximum: int,
    reserve_fraction: float,
    same_document_attempts: int,
    max_attempts_per_slot: int,
    max_total_attempt_multiplier: float,
    chunk_size: int,
    chunk_overlap: int,
    task_values: list[str],
    difficulty_values: list[str],
    generator_plugin: str,
    model: str,
    generator_params: list[str],
    judge_model: str | None,
    judge_plugin: str | None,
    judge_params: list[str],
    audit_fraction: float,
    formats: str,
    containers: str,
    train_split: float,
    validation_split: float,
    test_split: float,
    store_model_io: bool,
    fail_on_partial: bool,
) -> ProjectConfig:
    if document_count is not None and selection_fraction is not None:
        raise typer.BadParameter("use only one of --documents or --selection-fraction")
    selection = (
        CorpusSelection(count=document_count, seed=seed)
        if document_count is not None
        else CorpusSelection(fraction=selection_fraction if selection_fraction is not None else 1.0, seed=seed)
    )
    if source_plugin == "huggingface":
        source_params: dict[str, Any] = {
            "dataset": dataset,
            "split": split,
            "streaming": streaming,
            "cache_dir": str(cache_dir),
        }
        if subset is not None:
            source_params["subset"] = subset
    elif source_plugin == "local":
        source_params = {"path": dataset}
        if source_format is not None:
            source_params["format"] = source_format
    else:
        raise typer.BadParameter("--source must be 'huggingface' or 'local'")

    generation = GenerationConfig(
        plugin=generator_plugin,
        model=model,
        params=_key_values(generator_params, "--generator-param"),
    )
    task_weights = _weights(task_values, DEFAULT_TASKS, "--task")
    unknown_tasks = sorted(set(task_weights) - set(TASK_INSTRUCTIONS))
    if unknown_tasks:
        raise typer.BadParameter(f"unknown --task value(s): {', '.join(unknown_tasks)}")
    llm = None
    if judge_model is not None:
        llm = GenerationConfig(
            plugin=judge_plugin or generator_plugin,
            model=judge_model,
            params=_key_values(judge_params, "--judge-param"),
        )
    return ProjectConfig(
        name=name or _project_name(dataset),
        language=language,
        profile=profile,
        source=SourceConfig(
            plugin=source_plugin,
            params=source_params,
            field_map={
                "id": id_field,
                "text": text_field,
                "title": title_field,
                "sections": sections_field,
                "license": license_field,
            },
        ),
        selection=selection,
        target=TargetConfig(
            examples=examples,
            per_document={"minimum": per_document_minimum, "maximum": per_document_maximum},
            reserve_fraction=reserve_fraction,
            same_document_attempts=same_document_attempts,
            max_attempts_per_slot=max_attempts_per_slot,
            max_total_attempt_multiplier=max_total_attempt_multiplier,
            chunk_size_characters=chunk_size,
            chunk_overlap_characters=chunk_overlap,
        ),
        composition=CompositionConfig(
            tasks=DistributionConfig(weights=task_weights),
            difficulties=DistributionConfig(
                weights=_weights(difficulty_values, DEFAULT_DIFFICULTIES, "--difficulty")
            ),
        ),
        generation=generation,
        evaluation=EvaluationConfig(
            llm=llm,
            routing={"audit_fraction": audit_fraction},
        ),
        output=OutputConfig(
            formats=_csv(formats),
            containers=_csv(containers),
            splits=SplitConfig(train=train_split, validation=validation_split, test=test_split),
        ),
        runtime=RuntimeConfig(
            run_root=Path("runs"),
            cache_dir=cache_dir,
            store_model_io=store_model_io,
            fail_on_partial=fail_on_partial,
        ),
    )


def _doctor_table(report: dict) -> None:
    gpu_table = Table(title="GPUs")
    for column in ("Index", "Name", "Total MB", "Free MB", "Driver"):
        gpu_table.add_column(column)
    for gpu in report["gpus"]:
        gpu_table.add_row(
            str(gpu["index"]), gpu["name"], str(gpu["memory_total_mb"]), str(gpu["memory_free_mb"]), gpu["driver"]
        )
    console.print(gpu_table)
    package_table = Table(title="Runtime packages")
    package_table.add_column("Package")
    package_table.add_column("Available")
    for name, available in report["packages"].items():
        package_table.add_row(name, "yes" if available else "no")
    console.print(package_table)
    for error in report["errors"]:
        console.print(f"[red]Error:[/red] {error}")
    for warning in report["warnings"]:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
    console.print("[green]Ready[/green]" if report["ready"] else "[red]Not ready[/red]")


@app.command("schema")
def schema_command(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("sft-project.schema.json"),
) -> None:
    """Write the JSON Schema for project configurations."""
    write_config_schema(output)
    console.print(f"[green]Schema written:[/green] {output}")


@app.command("validate")
def validate_command(config: Annotated[Path, typer.Option("--config", "-c")]) -> None:
    """Validate a project configuration without creating artifacts."""
    value = load_config(config)
    console.print(f"[green]Valid configuration[/green] {value.name} ({value.config_hash})")


@app.command("plugins")
def plugins_command() -> None:
    """List built-in and installed extension plugins."""
    table = Table(title="Plugins")
    table.add_column("Kind")
    table.add_column("Names")
    for kind, names in available_plugins().items():
        table.add_row(kind, ", ".join(names))
    console.print(table)


@app.command()
def doctor(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    smoke_models: Annotated[bool, typer.Option("--smoke-models", help="Load configured models sequentially")] = False,
) -> None:
    """Inspect plugins, storage, GPUs, dependencies, and optionally model loading."""
    value = load_config(config) if config else None
    report = collect_doctor_report(value, smoke_models=smoke_models)
    _doctor_table(report)
    if not report["ready"]:
        raise typer.Exit(3)


@app.command("tune")
def tune_command(
    config: Annotated[Path, typer.Option("--config", "-c")],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("sft-project.tuned.json"),
    stage: Annotated[str, typer.Option("--stage", help="generation, evaluation, or both")] = "both",
    samples: Annotated[int, typer.Option("--samples", min=2)] = 32,
) -> None:
    """Benchmark serial and async profiles and write a reproducible tuned configuration."""
    if stage not in {"generation", "evaluation", "both"}:
        raise typer.BadParameter("--stage must be generation, evaluation, or both")
    value = load_config(config)
    tuned, report_path = tune_project(value, output, stage=stage, samples=samples)
    table = Table(title="Selected batching profiles")
    table.add_column("Stage")
    table.add_column("Mode")
    table.add_column("Inflight")
    table.add_row("generation", tuned.generation.batching.mode, str(tuned.generation.batching.max_inflight_requests))
    if tuned.evaluation.llm is not None:
        table.add_row(
            "evaluation",
            tuned.evaluation.llm.batching.mode,
            str(tuned.evaluation.llm.batching.max_inflight_requests),
        )
    console.print(table)
    console.print(f"[green]Tuned configuration:[/green] {output}")
    console.print(f"[green]Benchmark report:[/green] {report_path}")


@app.command("plan")
def plan_command(
    config: Annotated[Path, typer.Option("--config", "-c")],
    run_dir: Annotated[Path | None, typer.Option("--run-dir")] = None,
) -> None:
    """Scan the corpus and create an immutable execution plan."""
    value = load_config(config)
    with console.status("Scanning source and allocating slots..."):
        plan = build_plan(value, run_dir=run_dir)
    root = Path(plan.corpus_snapshot).parent
    table = Table(title="Dataset plan")
    table.add_column("Run")
    table.add_column("Documents")
    table.add_column("Reserve")
    table.add_column("Examples")
    table.add_row(
        plan.run_id,
        str(plan.estimates["selected_documents"]),
        str(plan.estimates["reserve_documents"]),
        str(len(plan.slots)),
    )
    console.print(table)
    console.print(f"[green]Plan written:[/green] {root / 'plan.json'}")


@app.command("run")
def run_command(
    dataset: Annotated[
        str | None,
        typer.Option("--dataset", help="Hugging Face dataset id or local corpus path"),
    ] = None,
    examples: Annotated[int | None, typer.Option("--examples", min=1, help="Final SFT example count")] = None,
    name: Annotated[str | None, typer.Option("--name", help="Project name; derived from the dataset by default")] = None,
    language: Annotated[str, typer.Option("--language")] = "en",
    source_plugin: Annotated[str, typer.Option("--source", help="huggingface or local")] = "huggingface",
    subset: Annotated[str | None, typer.Option("--subset")] = None,
    split: Annotated[str, typer.Option("--split")] = "train",
    streaming: Annotated[bool, typer.Option("--streaming/--no-streaming")] = True,
    source_format: Annotated[str | None, typer.Option("--source-format", help="json, jsonl, or parquet")] = None,
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path(".cache/sft-dataset-creator"),
    id_field: Annotated[str, typer.Option("--id-field")] = "id",
    text_field: Annotated[str, typer.Option("--text-field")] = "text",
    title_field: Annotated[str, typer.Option("--title-field")] = "title",
    sections_field: Annotated[str, typer.Option("--sections-field")] = "sections",
    license_field: Annotated[str, typer.Option("--license-field")] = "license",
    document_count: Annotated[int | None, typer.Option("--documents", min=1)] = None,
    selection_fraction: Annotated[float | None, typer.Option("--selection-fraction", min=0.0, max=1.0)] = None,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    profile: Annotated[str | None, typer.Option("--profile")] = None,
    per_document_minimum: Annotated[int, typer.Option("--per-document-min", min=0)] = 1,
    per_document_maximum: Annotated[int, typer.Option("--per-document-max", min=1)] = 3,
    reserve_fraction: Annotated[float, typer.Option("--reserve-fraction", min=0.0, max=0.99)] = 0.25,
    same_document_attempts: Annotated[int, typer.Option("--same-document-attempts", min=1)] = 2,
    max_attempts_per_slot: Annotated[int, typer.Option("--max-attempts", min=1)] = 3,
    max_total_attempt_multiplier: Annotated[float, typer.Option("--attempt-multiplier", min=1.0)] = 2.0,
    chunk_size: Annotated[int, typer.Option("--chunk-size", min=500)] = 8_000,
    chunk_overlap: Annotated[int, typer.Option("--chunk-overlap", min=0)] = 400,
    task_values: Annotated[
        list[str],
        typer.Option("--task", help="Repeat NAME=WEIGHT; defaults to the built-in task mix"),
    ] = [],
    difficulty_values: Annotated[
        list[str],
        typer.Option("--difficulty", help="Repeat NAME=WEIGHT; defaults to easy/medium/hard"),
    ] = [],
    generator_plugin: Annotated[str, typer.Option("--generator-plugin")] = "vllm_local",
    model: Annotated[str, typer.Option("--model")] = DEFAULT_MODEL,
    generator_params: Annotated[
        list[str],
        typer.Option("--generator-param", help="Repeat backend-specific KEY=VALUE"),
    ] = [],
    judge_model: Annotated[str | None, typer.Option("--judge-model", help="Enable selective LLM evaluation")] = None,
    judge_plugin: Annotated[str | None, typer.Option("--judge-plugin")] = None,
    judge_params: Annotated[
        list[str],
        typer.Option("--judge-param", help="Repeat judge backend KEY=VALUE"),
    ] = [],
    audit_fraction: Annotated[float, typer.Option("--audit-fraction", min=0.0, max=1.0)] = 0.10,
    formats: Annotated[str, typer.Option("--formats", help="Comma-separated output views")] = "messages,prompt_completion,alpaca",
    containers: Annotated[str, typer.Option("--containers", help="Comma-separated jsonl/parquet containers")] = "jsonl",
    train_split: Annotated[float, typer.Option("--train-split", min=0.0, max=1.0)] = 0.90,
    validation_split: Annotated[float, typer.Option("--validation-split", min=0.0, max=1.0)] = 0.05,
    test_split: Annotated[float, typer.Option("--test-split", min=0.0, max=1.0)] = 0.05,
    store_model_io: Annotated[bool, typer.Option("--store-model-io/--no-store-model-io")] = True,
    fail_on_partial: Annotated[bool, typer.Option("--fail-on-partial/--allow-partial")] = True,
    plan: Annotated[Path | None, typer.Option("--plan", "-p")] = None,
    run_dir: Annotated[Path | None, typer.Option("--run-dir")] = None,
    resume: Annotated[Path | None, typer.Option("--resume", help="Existing run directory")] = None,
) -> None:
    """Build and execute a project directly from CLI options, or resume a run."""
    supplied = sum((dataset is not None, plan is not None, resume is not None))
    if supplied != 1:
        raise typer.BadParameter("provide exactly one of --dataset, --plan, or --resume")
    if dataset is not None:
        if examples is None:
            raise typer.BadParameter("--examples is required with --dataset")
        value = _direct_config(
            dataset=dataset,
            examples=examples,
            name=name,
            language=language,
            source_plugin=source_plugin,
            subset=subset,
            split=split,
            streaming=streaming,
            source_format=source_format,
            cache_dir=cache_dir,
            id_field=id_field,
            text_field=text_field,
            title_field=title_field,
            sections_field=sections_field,
            license_field=license_field,
            document_count=document_count,
            selection_fraction=selection_fraction,
            seed=seed,
            profile=profile,
            per_document_minimum=per_document_minimum,
            per_document_maximum=per_document_maximum,
            reserve_fraction=reserve_fraction,
            same_document_attempts=same_document_attempts,
            max_attempts_per_slot=max_attempts_per_slot,
            max_total_attempt_multiplier=max_total_attempt_multiplier,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            task_values=task_values,
            difficulty_values=difficulty_values,
            generator_plugin=generator_plugin,
            model=model,
            generator_params=generator_params,
            judge_model=judge_model,
            judge_plugin=judge_plugin,
            judge_params=judge_params,
            audit_fraction=audit_fraction,
            formats=formats,
            containers=containers,
            train_split=train_split,
            validation_split=validation_split,
            test_split=test_split,
            store_model_io=store_model_io,
            fail_on_partial=fail_on_partial,
        )
        dataset_plan = build_plan(value, run_dir=run_dir)
        root = Path(dataset_plan.corpus_snapshot).parent
    elif plan:
        dataset_plan = load_plan(plan)
        root = Path(plan).parent
        value = load_config(root / "config.resolved.json")
    else:
        root = Path(resume)
        dataset_plan = load_plan(root / "plan.json")
        value = load_config(root / "config.resolved.json")
    report = collect_doctor_report(value)
    if not report["ready"]:
        _doctor_table(report)
        raise typer.Exit(3)
    attach_environment(root, report)
    progress = Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console)
    task_id = progress.add_task("Starting run", total=None)

    def callback(kind: str, payload: dict) -> None:
        if kind == "generation_started":
            progress.update(task_id, description=f"Generating round {payload['round']} ({payload['slots']} slots)")
        elif kind == "evaluation_started":
            progress.update(task_id, description=f"Evaluating {payload['candidates']} routed candidates")
        elif kind == "round_finished":
            progress.update(task_id, description=f"Round {payload['round']}: {payload['accepted']}/{payload['target']} accepted")

    with progress:
        result = execute_plan(dataset_plan, value, run_dir=root, callback=callback)
    console.print(Panel.fit(result.model_dump_json(indent=2), title="Run report"))
    if result.status == "partial" and value.runtime.fail_on_partial:
        raise typer.Exit(4)


@app.command("status")
def status_command(run_dir: Annotated[Path, typer.Argument()]) -> None:
    """Show resumable run counters and the latest report."""
    with RunState(run_dir / "run.db") as state:
        counts = state.counts()
        deficits = state.deficits()
    table = Table(title=f"Run status: {run_dir.name}")
    table.add_column("Target")
    table.add_column("Accepted")
    table.add_column("Attempted")
    table.add_column("Rejected")
    table.add_column("Speculative")
    table.add_row(*(str(counts[key]) for key in ("target", "accepted", "attempted", "rejected", "speculative")))
    console.print(table)
    if deficits:
        console.print_json(json.dumps(deficits))


@app.command("inspect")
def inspect_command(
    run_dir: Annotated[Path, typer.Argument()],
    limit: Annotated[int, typer.Option("--limit", "-n", min=1)] = 10,
) -> None:
    """Inspect accepted examples from a run."""
    with RunState(run_dir / "run.db") as state:
        candidates = list(state.accepted_candidates())[:limit]
    for candidate in candidates:
        console.print(
            Panel(
                f"[bold]{candidate.instruction}[/bold]\n\n{candidate.output}",
                title=f"{candidate.id} | {candidate.task} | {candidate.difficulty}",
            )
        )


@app.command("export")
def export_command(run_dir: Annotated[Path, typer.Argument()]) -> None:
    """Regenerate configured output views from accepted canonical records."""
    target = export_run(run_dir)
    console.print(f"[green]Exports written:[/green] {target}")


@app.command("audit-sample")
def audit_sample_command(
    run_dir: Annotated[Path, typer.Argument()],
    size: Annotated[int, typer.Option("--size", min=1)] = 300,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Create a blinded, stratified manual-review sample and a separate answer key."""
    path = create_audit_sample(run_dir, size=size, seed=seed)
    console.print(f"[green]Audit sample written:[/green] {path}")


@app.command("audit-score")
def audit_score_command(run_dir: Annotated[Path, typer.Argument()]) -> None:
    """Score a completed manual audit against gates and selective LLM routing."""
    report = score_audit(run_dir)
    console.print(Panel.fit(json.dumps(report, indent=2), title="Audit report"))


@app.command("publish")
def publish_command(
    run_dir: Annotated[Path, typer.Argument()],
    repo_id: Annotated[str | None, typer.Option("--repo-id")] = None,
    public: Annotated[bool, typer.Option("--public", help="Publish publicly instead of the private default")] = False,
) -> None:
    """Publish a completed generic export to the Hugging Face Hub."""
    report_path = run_dir / "report.json"
    if report_path.exists() and json.loads(report_path.read_text())["status"] != "completed":
        raise typer.BadParameter("partial runs cannot be published without completing the target")
    url = publish_run(run_dir, repo_id=repo_id, private=not public)
    console.print(f"[green]Published:[/green] {url}")
