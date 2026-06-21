from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from sft_dataset_creator.audit import create_audit_sample, score_audit
from sft_dataset_creator.config import load_config, write_config_schema
from sft_dataset_creator.doctor import attach_environment, collect_doctor_report
from sft_dataset_creator.engine import execute_plan
from sft_dataset_creator.exporters import export_run
from sft_dataset_creator.planner import build_plan, load_plan
from sft_dataset_creator.publisher import publish_run
from sft_dataset_creator.registry import available_plugins
from sft_dataset_creator.state import RunState
from sft_dataset_creator.tuning import tune_project
from sft_dataset_creator.wizard import run_wizard


app = typer.Typer(
    name="sft-dataset",
    help="Plan, generate, evaluate, and publish reproducible synthetic SFT datasets.",
    no_args_is_help=True,
)
console = Console()


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


@app.command()
def wizard(
    output: Annotated[Path, typer.Option("--output", "-o", help="Configuration JSON path")] = Path("sft-project.json"),
) -> None:
    """Create a validated project configuration interactively."""
    config = run_wizard(output)
    console.print(Panel.fit(f"Configuration written to [bold]{output}[/bold]\nHash: {config.config_hash}"))


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
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    plan: Annotated[Path | None, typer.Option("--plan", "-p")] = None,
    run_dir: Annotated[Path | None, typer.Option("--run-dir")] = None,
    resume: Annotated[Path | None, typer.Option("--resume", help="Existing run directory")] = None,
) -> None:
    """Execute a config, immutable plan, or interrupted run without interactive prompts."""
    supplied = sum(item is not None for item in (config, plan, resume))
    if supplied != 1:
        raise typer.BadParameter("provide exactly one of --config, --plan, or --resume")
    if config:
        value = load_config(config)
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
