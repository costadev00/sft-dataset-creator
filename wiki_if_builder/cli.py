from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from wiki_if_builder.analyst import (
    analyze_article,
    build_synthetic_analyst_output,
    error_record,
)
from wiki_if_builder.config import AppConfig, load_config
from wiki_if_builder.dataset_loader import (
    iter_articles,
    load_wikipedia_dataset,
    synthetic_dry_run_articles,
)
from wiki_if_builder.exporter import export_all_from_intermediate
from wiki_if_builder.hf_publisher import publish_dataset
from wiki_if_builder.judge import heuristic_judge_candidate, judge_candidate
from wiki_if_builder.llm_client import RoundRobinLLMClient, build_llm_client
from wiki_if_builder.normalizer import normalize_analyst_output
from wiki_if_builder.schemas import AnalystOutput, JudgeResult, TriageResult
from wiki_if_builder.triage import is_valid_for_llm, triage_article
from wiki_if_builder.utils import (
    JsonlWriter,
    get_disk_report,
    get_gpu_info,
    get_ram_info,
    iter_jsonl,
    normalize_page_id,
    parse_bool,
    safe_size,
)


app = typer.Typer(help="Pipeline local para labeling documental e dataset Instruction Following.")
console = Console()


INTERMEDIATE_FILENAMES = {
    "triage": "triage_report.jsonl",
    "analyst": "analyst_outputs.jsonl",
    "normalized": "normalized_outputs.jsonl",
    "judge": "judge_results.jsonl",
    "rejected": "rejected_candidates.jsonl",
    "errors": "raw_errors.jsonl",
}


def _storage_overrides(
    *,
    work_dir: str | None = None,
    cache_dir: str | None = None,
    output_dir: str | None = None,
    tmp_dir: str | None = None,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    inferred_work_dir: Path | None = None
    if not work_dir:
        if cache_dir:
            inferred_work_dir = Path(cache_dir).parent
        elif output_dir:
            inferred_work_dir = Path(output_dir).parent
        elif tmp_dir:
            inferred_work_dir = Path(tmp_dir).parent
    if work_dir:
        work = Path(work_dir)
        overrides["work_dir"] = work
        if cache_dir is None:
            overrides["cache_dir"] = work / "cache"
        if output_dir is None:
            overrides["output_dir"] = work / "outputs"
        overrides["model_cache_dir"] = work / "models"
    elif inferred_work_dir is not None:
        overrides["work_dir"] = inferred_work_dir
        overrides["model_cache_dir"] = inferred_work_dir / "models"
        if cache_dir is None:
            overrides["cache_dir"] = inferred_work_dir / "cache"
        if output_dir is None:
            overrides["output_dir"] = inferred_work_dir / "outputs"
    if cache_dir:
        cache = Path(cache_dir)
        overrides["cache_dir"] = cache
        overrides["hf_home"] = cache / "huggingface"
        overrides["hf_datasets_cache"] = cache / "huggingface" / "datasets"
        overrides["transformers_cache"] = cache / "huggingface" / "transformers"
    elif "cache_dir" in overrides:
        cache = Path(overrides["cache_dir"])
        overrides["hf_home"] = cache / "huggingface"
        overrides["hf_datasets_cache"] = cache / "huggingface" / "datasets"
        overrides["transformers_cache"] = cache / "huggingface" / "transformers"
    if output_dir:
        overrides["output_dir"] = Path(output_dir)
    if tmp_dir:
        overrides["tmp_dir"] = Path(tmp_dir)
    return overrides


def _intermediate_paths(config: AppConfig) -> dict[str, Path]:
    return {name: config.intermediate_dir / filename for name, filename in INTERMEDIATE_FILENAMES.items()}


def _existing_analyzed_page_ids(path: Path) -> set[int]:
    page_ids: set[int] = set()
    for row in iter_jsonl(path):
        source = row.get("source") or {}
        if "page_id" in source:
            page_ids.add(normalize_page_id(source["page_id"]))
    return page_ids


def collect_doctor_report(config: AppConfig) -> dict[str, Any]:
    storage = config.validate_storage()
    return {
        "gpus": get_gpu_info(),
        "ram": get_ram_info(),
        "disks": get_disk_report(["/", "/mnt/disco1", "/mnt/disco2"]),
        "paths": {
            "WORK_DIR": str(config.work_dir),
            "CACHE_DIR": str(config.cache_dir),
            "OUTPUT_DIR": str(config.output_dir),
            "HF_HOME": str(config.hf_home),
            "HF_DATASETS_CACHE": str(config.hf_datasets_cache),
            "TMPDIR": str(config.tmp_dir),
        },
        "storage": {
            "ready": storage.ready,
            "warnings": storage.warnings,
            "errors": storage.errors,
            "free_gb": storage.free_gb,
        },
    }


def _print_doctor_report(report: dict[str, Any]) -> None:
    console.print("[bold]wiki-if-builder doctor[/bold]")
    gpus = report["gpus"]
    if gpus:
        table = Table(title="GPUs detectadas")
        table.add_column("Index")
        table.add_column("Nome")
        table.add_column("Memória total")
        for gpu in gpus:
            table.add_row(str(gpu["index"]), str(gpu["name"]), f"{gpu['memory_total_mb']} MB")
        console.print(table)
    else:
        console.print("[yellow]Nenhuma GPU detectada via nvidia-smi.[/yellow]")

    ram = report["ram"]
    console.print(f"RAM total: {ram['total_gb']:.1f} GB | disponível: {ram['available_gb']:.1f} GB")

    disk_table = Table(title="Discos")
    disk_table.add_column("Path")
    disk_table.add_column("Existe")
    disk_table.add_column("Livre")
    disk_table.add_column("Uso")
    for path, info in report["disks"].items():
        disk_table.add_row(
            path,
            "sim" if info["exists"] else "não",
            f"{info['free_gb']:.1f} GB",
            f"{info['percent_used']:.1f}%",
        )
    console.print(disk_table)

    path_table = Table(title="Paths efetivos")
    path_table.add_column("Variável")
    path_table.add_column("Valor")
    for key, value in report["paths"].items():
        path_table.add_row(key, value)
    console.print(path_table)

    storage = report["storage"]
    for warning in storage["warnings"]:
        console.print(f"[yellow]Aviso:[/yellow] {warning}")
    for error in storage["errors"]:
        console.print(f"[red]Erro:[/red] {error}")
    console.print("[green]Ambiente pronto para rodar a pipeline.[/green]" if storage["ready"] else "[red]Ambiente não está pronto.[/red]")


@app.command()
def doctor(
    work_dir: str | None = typer.Option(None, "--work-dir"),
    cache_dir: str | None = typer.Option(None, "--cache-dir"),
    output_dir: str | None = typer.Option(None, "--output-dir"),
    tmp_dir: str | None = typer.Option(None, "--tmp-dir"),
    min_free_output_gb: float = typer.Option(50.0, "--min-free-output-gb"),
    min_free_cache_gb: float = typer.Option(100.0, "--min-free-cache-gb"),
) -> None:
    """Verifica GPUs, RAM, discos e paths efetivos."""
    config = load_config(
        **_storage_overrides(work_dir=work_dir, cache_dir=cache_dir, output_dir=output_dir, tmp_dir=tmp_dir),
        min_free_output_gb=min_free_output_gb,
        min_free_cache_gb=min_free_cache_gb,
    )
    report = collect_doctor_report(config)
    _print_doctor_report(report)
    if not report["storage"]["ready"]:
        raise typer.Exit(code=1)


def _process_valid_article(
    article: dict[str, Any],
    triage_result: TriageResult,
    config: AppConfig,
    llm_client: RoundRobinLLMClient | None,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    try:
        if config.dry_run:
            analyst_output = build_synthetic_analyst_output(article, triage_result, config)
        else:
            if llm_client is None:
                raise RuntimeError("LLM client não inicializado")
            analyst_output = analyze_article(article, triage_result, config, llm_client)
    except Exception as exc:  # noqa: BLE001 - pipeline não deve parar por um artigo ruim
        return {"errors": [error_record("analyst", article, exc)]}

    normalized = normalize_analyst_output(analyst_output)
    judge_results: list[JudgeResult] = []
    rejected: list[dict[str, Any]] = []
    for candidate in normalized.if_candidates:
        try:
            if config.dry_run:
                judge_result = heuristic_judge_candidate(candidate, article, config)
            else:
                judge_result = judge_candidate(candidate, article, config, llm_client)
        except Exception as exc:  # noqa: BLE001
            errors.append(error_record("judge", article, exc))
            fallback_config = config.model_copy(update={"enable_judge": False})
            judge_result = heuristic_judge_candidate(candidate, article, fallback_config)
            judge_result = judge_result.model_copy(
                update={
                    "judge": judge_result.judge.model_copy(
                        update={
                            "verdict": "review",
                            "issues": list(dict.fromkeys(judge_result.judge.issues + ["judge_error"])),
                            "needs_human_review": True,
                        }
                    )
                }
            )
        judge_results.append(judge_result)
        if judge_result.judge.verdict != "accept":
            rejected.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "source_page_id": normalized.source.page_id,
                    "source_title": normalized.source.title,
                    "judge_verdict": judge_result.judge.verdict,
                    "issues": judge_result.judge.issues,
                }
            )
    return {
        "analyst": analyst_output,
        "normalized": normalized,
        "judge_results": judge_results,
        "rejected": rejected,
        "errors": errors,
        "llm_called": not config.dry_run,
    }


def _write_processed_result(result: dict[str, Any], writers: dict[str, JsonlWriter], stats: dict[str, int]) -> None:
    for error in result.get("errors", []):
        writers["errors"].write(error)
    analyst_output: AnalystOutput | None = result.get("analyst")
    normalized: AnalystOutput | None = result.get("normalized")
    if analyst_output is not None:
        writers["analyst"].write(analyst_output.model_dump(mode="json"))
        stats["sent_llm"] += 1 if result.get("llm_called") else 0
    if normalized is not None:
        writers["normalized"].write(normalized.model_dump(mode="json"))
        stats["candidates_generated"] += len(normalized.if_candidates)
    for judge_result in result.get("judge_results", []):
        writers["judge"].write(judge_result.model_dump(mode="json"))
        verdict = judge_result.judge.verdict
        if verdict == "accept":
            stats["candidates_accepted"] += 1
        elif verdict == "reject":
            stats["candidates_rejected"] += 1
        else:
            stats["candidates_review"] += 1
    for rejected in result.get("rejected", []):
        writers["rejected"].write(rejected)


def _drain_futures(
    pending: set[Future],
    writers: dict[str, JsonlWriter],
    stats: dict[str, int],
    *,
    wait_for_first: bool,
) -> set[Future]:
    if not pending:
        return pending
    if wait_for_first:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
    else:
        done, pending = wait(pending)
    for future in done:
        _write_processed_result(future.result(), writers, stats)
    return set(pending)


def _open_intermediate_writers(config: AppConfig) -> dict[str, JsonlWriter]:
    paths = _intermediate_paths(config)
    return {
        key: JsonlWriter(path, append=config.resume)
        for key, path in paths.items()
    }


@app.command("run")
def run_pipeline(
    dataset_name: str | None = typer.Option(None, "--dataset-name"),
    split: str = typer.Option("train", "--split"),
    max_articles: int | None = typer.Option(None, "--max-articles"),
    work_dir: str | None = typer.Option(None, "--work-dir"),
    cache_dir: str | None = typer.Option(None, "--cache-dir"),
    output_dir: str | None = typer.Option(None, "--output-dir"),
    tmp_dir: str | None = typer.Option(None, "--tmp-dir"),
    enable_judge: str = typer.Option("false", "--enable-judge"),
    max_input_chars: int = typer.Option(120_000, "--max-input-chars"),
    max_output_tokens: int = typer.Option(4096, "--max-output-tokens"),
    candidates_per_article: int = typer.Option(3, "--candidates-per-article"),
    dry_run: str = typer.Option("false", "--dry-run"),
    resume: str = typer.Option("false", "--resume"),
    include_review: str = typer.Option("false", "--include-review"),
    private: str = typer.Option("true", "--private"),
    num_workers: int = typer.Option(1, "--num-workers"),
    max_concurrent_llm_calls: int | None = typer.Option(None, "--max-concurrent-llm-calls"),
    min_free_output_gb: float = typer.Option(50.0, "--min-free-output-gb"),
    min_free_cache_gb: float = typer.Option(100.0, "--min-free-cache-gb"),
) -> None:
    """Executa triagem, análise LLM, normalização, judge opcional e export."""
    config = load_config(
        **_storage_overrides(work_dir=work_dir, cache_dir=cache_dir, output_dir=output_dir, tmp_dir=tmp_dir),
        dataset_name=dataset_name,
        split=split,
        max_articles=max_articles,
        enable_judge=parse_bool(enable_judge),
        max_input_chars=max_input_chars,
        max_output_tokens=max_output_tokens,
        candidates_per_article=candidates_per_article,
        dry_run=parse_bool(dry_run),
        resume=parse_bool(resume),
        include_review=parse_bool(include_review),
        private=parse_bool(private),
        num_workers=max(1, num_workers),
        max_concurrent_llm_calls=max_concurrent_llm_calls,
        min_free_output_gb=min_free_output_gb,
        min_free_cache_gb=min_free_cache_gb,
    )
    storage = config.validate_storage()
    for warning in storage.warnings:
        console.print(f"[yellow]Aviso:[/yellow] {warning}")
    if storage.errors:
        for error in storage.errors:
            console.print(f"[red]Erro:[/red] {error}")
        raise typer.Exit(code=1)

    console.print(f"WORK_DIR={config.work_dir}")
    console.print(f"CACHE_DIR={config.cache_dir}")
    console.print(f"OUTPUT_DIR={config.output_dir}")
    console.print(f"TMPDIR={config.tmp_dir}")

    paths = _intermediate_paths(config)
    done_page_ids = _existing_analyzed_page_ids(paths["analyst"]) if config.resume else set()
    if done_page_ids:
        console.print(f"[cyan]Resume ativo:[/cyan] {len(done_page_ids)} page_id(s) já presentes em analyst_outputs.jsonl")

    llm_client = None if config.dry_run else build_llm_client(config)
    if config.dry_run:
        console.print("[cyan]Dry-run ativo:[/cyan] usando artigos sintéticos e judge heurístico, sem chamada LLM.")
        source_articles = synthetic_dry_run_articles(config.max_articles)
    else:
        dataset = load_wikipedia_dataset(
            config.dataset_name,
            split=config.split,
            cache_dir=str(config.hf_datasets_cache),
            streaming=True,
        )
        source_articles = iter_articles(dataset, config.max_articles)

    stats = {
        "read": 0,
        "triage_rejected": 0,
        "sent_llm": 0,
        "candidates_generated": 0,
        "candidates_accepted": 0,
        "candidates_rejected": 0,
        "candidates_review": 0,
    }

    writer_objects = _open_intermediate_writers(config)
    with writer_objects["triage"] as triage_writer, writer_objects["analyst"] as analyst_writer, writer_objects[
        "normalized"
    ] as normalized_writer, writer_objects["judge"] as judge_writer, writer_objects[
        "rejected"
    ] as rejected_writer, writer_objects[
        "errors"
    ] as errors_writer:
        writers = {
            "triage": triage_writer,
            "analyst": analyst_writer,
            "normalized": normalized_writer,
            "judge": judge_writer,
            "rejected": rejected_writer,
            "errors": errors_writer,
        }
        pending: set[Future] = set()
        pending_limit = max(1, config.num_workers * 2)
        with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
            for article in source_articles:
                page_id = normalize_page_id(article.get("page_id"))
                if config.resume and page_id in done_page_ids:
                    continue
                stats["read"] += 1
                triage_result = triage_article(article)
                writers["triage"].write(
                    {
                        "page_id": page_id,
                        "title": str(article.get("title") or ""),
                        **triage_result.model_dump(mode="json"),
                    }
                )
                if not is_valid_for_llm(triage_result):
                    stats["triage_rejected"] += 1
                    continue
                pending.add(executor.submit(_process_valid_article, article, triage_result, config, llm_client))
                if len(pending) >= pending_limit:
                    pending = _drain_futures(pending, writers, stats, wait_for_first=True)
                if stats["read"] % max(1, config.log_every) == 0:
                    console.print(
                        f"lidos={stats['read']} rejeitados_triage={stats['triage_rejected']} "
                        f"candidatos={stats['candidates_generated']} aceitos={stats['candidates_accepted']}"
                    )
            pending = _drain_futures(pending, writers, stats, wait_for_first=False)

    labels_dir, if_dir = export_all_from_intermediate(
        config.output_dir,
        include_review=config.include_review,
        model_name=config.model_name,
        judge_model_name=config.judge_model_name,
        pipeline_version=config.pipeline_version,
    )
    console.print("[bold green]Pipeline concluída.[/bold green]")
    console.print(str(stats))
    console.print(f"Dataset labels: {labels_dir} ({safe_size(labels_dir)} bytes)")
    console.print(f"Dataset IF: {if_dir} ({safe_size(if_dir)} bytes)")


@app.command()
def triage(
    dataset_name: str | None = typer.Option(None, "--dataset-name"),
    split: str = typer.Option("train", "--split"),
    max_articles: int | None = typer.Option(None, "--max-articles"),
    work_dir: str | None = typer.Option(None, "--work-dir"),
    cache_dir: str | None = typer.Option(None, "--cache-dir"),
    output_dir: str | None = typer.Option(None, "--output-dir"),
    tmp_dir: str | None = typer.Option(None, "--tmp-dir"),
) -> None:
    """Executa somente a triagem determinística e salva triage_report.jsonl."""
    config = load_config(
        **_storage_overrides(work_dir=work_dir, cache_dir=cache_dir, output_dir=output_dir, tmp_dir=tmp_dir),
        dataset_name=dataset_name,
        split=split,
        max_articles=max_articles,
    )
    dataset = load_wikipedia_dataset(
        config.dataset_name,
        split=config.split,
        cache_dir=str(config.hf_datasets_cache),
        streaming=True,
    )
    report_path = config.intermediate_dir / "triage_report.jsonl"
    count = 0
    rejected = 0
    with JsonlWriter(report_path, append=False) as writer:
        for article in iter_articles(dataset, config.max_articles):
            result = triage_article(article)
            if not is_valid_for_llm(result):
                rejected += 1
            writer.write(
                {
                    "page_id": normalize_page_id(article.get("page_id")),
                    "title": str(article.get("title") or ""),
                    **result.model_dump(mode="json"),
                }
            )
            count += 1
            if count % max(1, config.log_every) == 0:
                console.print(f"triage lidos={count} rejeitados={rejected}")
    console.print(f"[green]triage_report salvo em {report_path}[/green]")


@app.command("export")
def export_cmd(
    output_dir: str | None = typer.Option(None, "--output-dir"),
    include_review: str = typer.Option("false", "--include-review"),
) -> None:
    """Reprocessa intermediários e gera os dois datasets finais."""
    config = load_config(
        **_storage_overrides(output_dir=output_dir),
        include_review=parse_bool(include_review),
    )
    labels_dir, if_dir = export_all_from_intermediate(
        config.output_dir,
        include_review=config.include_review,
        model_name=config.model_name,
        judge_model_name=config.judge_model_name,
        pipeline_version=config.pipeline_version,
    )
    console.print(f"[green]Export labels:[/green] {labels_dir}")
    console.print(f"[green]Export IF:[/green] {if_dir}")


@app.command()
def inspect(
    output_dir: str | None = typer.Option(None, "--output-dir"),
    limit: int = typer.Option(10, "--limit"),
) -> None:
    """Mostra exemplos aceitos e rejeitados usando Rich."""
    config = load_config(**_storage_overrides(output_dir=output_dir))
    normalized_path = config.intermediate_dir / "normalized_outputs.jsonl"
    triage_path = config.intermediate_dir / "triage_report.jsonl"

    accepted_table = Table(title="Exemplos aceitos/normalizados")
    accepted_table.add_column("page_id")
    accepted_table.add_column("title")
    accepted_table.add_column("categoria")
    accepted_table.add_column("candidatos")
    shown = 0
    for row in iter_jsonl(normalized_path):
        source = row.get("source", {})
        labels = row.get("document_labels", {})
        accepted_table.add_row(
            str(source.get("page_id", "")),
            str(source.get("title", ""))[:80],
            str(labels.get("primary_category", "")),
            str(len(row.get("if_candidates", []))),
        )
        shown += 1
        if shown >= limit:
            break
    console.print(accepted_table)

    rejected_table = Table(title="Triagem rejeitada")
    rejected_table.add_column("page_id")
    rejected_table.add_column("title")
    rejected_table.add_column("status")
    rejected_table.add_column("reason")
    shown = 0
    for row in iter_jsonl(triage_path):
        if row.get("status") in {"valid_article", "valid_short_article"}:
            continue
        rejected_table.add_row(
            str(row.get("page_id", "")),
            str(row.get("title", ""))[:80],
            str(row.get("status", "")),
            str(row.get("reason", ""))[:120],
        )
        shown += 1
        if shown >= limit:
            break
    console.print(rejected_table)


@app.command("publish-labels")
def publish_labels(
    repo_id: str = typer.Option("costadev00/wikipedia-pt-br-article-labels-gemma", "--repo-id"),
    local_dir: str = typer.Option("/mnt/disco1/wiki-if-builder/outputs/document_labels", "--local-dir"),
    private: str = typer.Option("true", "--private"),
) -> None:
    publish_dataset(local_dir=local_dir, repo_id=repo_id, private=parse_bool(private))


@app.command("publish-if")
def publish_if(
    repo_id: str = typer.Option("costadev00/wikipedia-pt-br-instructions-gemma", "--repo-id"),
    local_dir: str = typer.Option("/mnt/disco1/wiki-if-builder/outputs/instruction_following", "--local-dir"),
    private: str = typer.Option("true", "--private"),
) -> None:
    publish_dataset(local_dir=local_dir, repo_id=repo_id, private=parse_bool(private))


@app.command("publish-all")
def publish_all(
    labels_repo_id: str = typer.Option(
        "costadev00/wikipedia-pt-br-article-labels-gemma", "--labels-repo-id"
    ),
    if_repo_id: str = typer.Option("costadev00/wikipedia-pt-br-instructions-gemma", "--if-repo-id"),
    output_dir: str = typer.Option("/mnt/disco1/wiki-if-builder/outputs", "--output-dir"),
    private: str = typer.Option("true", "--private"),
) -> None:
    private_bool = parse_bool(private)
    publish_dataset(
        local_dir=str(Path(output_dir) / "document_labels"),
        repo_id=labels_repo_id,
        private=private_bool,
    )
    publish_dataset(
        local_dir=str(Path(output_dir) / "instruction_following"),
        repo_id=if_repo_id,
        private=private_bool,
    )
