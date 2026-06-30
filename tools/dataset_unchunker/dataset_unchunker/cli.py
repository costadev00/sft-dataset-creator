from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from dataset_unchunker.grouping import collect_groups
from dataset_unchunker.loaders import iter_rows
from dataset_unchunker.publish import publish_output
from dataset_unchunker.reconstruct import load_tokenizer, reconstruct_group
from dataset_unchunker.reports import dataset_card, make_report, write_json, write_quarantine
from dataset_unchunker.schema import OutputSpec, QuarantineEntry, SourceSpec, SplitPolicy, Strategy, UnchunkConfig


app = typer.Typer(help="Reconstruct chunked datasets into full-document rows.")


@app.callback()
def main() -> None:
    """Generic dataset unchunking utilities."""


def _parse_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _write_parquet(path: Path, rows: list[dict[str, object]], text_column: str) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("writing parquet requires pyarrow") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        columns: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    columns.append(key)
        normalized = [{column: row.get(column) for column in columns} for row in rows]
        table = pa.Table.from_pylist(normalized)
    else:
        table = pa.table(
            {
                "reconstructed_group_id": pa.array([], type=pa.string()),
                text_column: pa.array([], type=pa.string()),
                "split": pa.array([], type=pa.string()),
                "source_chunk_count": pa.array([], type=pa.int64()),
                "reconstruction_status": pa.array([], type=pa.string()),
            }
        )
    pq.write_table(table, path)


@app.command("unchunk")
def unchunk_command(
    repo_id: Annotated[str | None, typer.Option("--repo-id", help="Hugging Face dataset repo id.")] = None,
    revision: Annotated[str | None, typer.Option("--revision", help="Pinned Hub revision.")] = None,
    config_name: Annotated[str | None, typer.Option("--config", help="Dataset config name.")] = None,
    splits: Annotated[str, typer.Option("--splits", help="Comma-separated splits to read.")] = "train",
    source_file: Annotated[
        str | None,
        typer.Option("--source-file", help="Local parquet path or parquet path inside --repo-id."),
    ] = None,
    text_column: Annotated[str, typer.Option("--text-column")] = "text",
    chunk_index_column: Annotated[str, typer.Option("--chunk-index-column")] = "chunk_index",
    chunk_total_column: Annotated[str, typer.Option("--chunk-total-column")] = "chunk_total",
    group_by: Annotated[str | None, typer.Option("--group-by", help="Comma-separated stable grouping columns.")] = None,
    group_id_column: Annotated[str | None, typer.Option("--group-id-column", help="Single stable group id column.")] = None,
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("output/unchunked-dataset"),
    output_repo_id: Annotated[str | None, typer.Option("--output-repo-id")] = None,
    public: Annotated[
        bool,
        typer.Option("--public/--private", help="Publish publicly instead of the private default."),
    ] = False,
    no_push: Annotated[bool, typer.Option("--no-push", help="Only write local files.")] = False,
    strategy: Annotated[Strategy, typer.Option("--strategy", case_sensitive=False)] = Strategy.AUTO,
    tokenizer_name: Annotated[str | None, typer.Option("--tokenizer", help="Tokenizer for token-overlap mode.")] = None,
    max_overlap: Annotated[int, typer.Option("--max-overlap", min=0)] = 256,
    expected_overlap: Annotated[int | None, typer.Option("--expected-overlap", min=0)] = None,
    split_policy: Annotated[SplitPolicy, typer.Option("--split-policy", case_sensitive=False)] = SplitPolicy.TRAIN,
    source_split_column: Annotated[str, typer.Option("--source-split-column")] = "split",
    drop_columns: Annotated[str, typer.Option("--drop-columns", help="Comma-separated extra chunk metadata columns.")] = "token_count",
    max_groups: Annotated[int | None, typer.Option("--max-groups", min=1)] = None,
    token_env: Annotated[str, typer.Option("--token-env")] = "HF_TOKEN",
) -> None:
    """Reconstruct full rows from chunked dataset rows."""
    config = UnchunkConfig(
        text_column=text_column,
        chunk_index_column=chunk_index_column,
        chunk_total_column=chunk_total_column,
        group_by=_parse_csv(group_by),
        group_id_column=group_id_column,
        tokenizer=tokenizer_name,
        strategy=strategy,
        max_overlap=max_overlap,
        expected_overlap=expected_overlap,
        split_policy=split_policy,
        source_split_column=source_split_column,
        drop_columns=_parse_csv(drop_columns),
        max_groups=max_groups,
    )
    try:
        config.validate()
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    source = SourceSpec(
        repo_id=repo_id,
        revision=revision,
        config_name=config_name,
        splits=_parse_csv(splits) or ("train",),
        source_file=source_file,
        token_env=token_env,
    )
    output = OutputSpec(
        output_dir=output_dir,
        output_repo_id=output_repo_id,
        private=not public,
        no_push=no_push,
        token_env=token_env,
    )

    typer.echo("Loading rows and grouping chunks...")
    groups, input_rows, quarantined = collect_groups(iter_rows(source), config)
    typer.echo(f"Grouped {input_rows} rows into {len(groups)} candidate documents.")

    tokenizer = load_tokenizer(config.tokenizer)
    reconstructed_rows: list[dict[str, object]] = []
    conflict_counter: Counter[str] = Counter()
    overlap_counter: Counter[int] = Counter()

    for group_id, rows in groups.items():
        result = reconstruct_group(group_id, rows, config, tokenizer)
        if isinstance(result, QuarantineEntry):
            quarantined.append(result)
            continue
        reconstructed_rows.append(result.row)
        conflict_counter.update(result.conflicts.keys())
        overlap_counter.update(result.inferred_overlaps)

    report = make_report(
        source=source,
        config=config,
        input_rows=input_rows,
        groups_seen=len(groups),
        reconstructed_groups=len(reconstructed_rows),
        quarantined_groups=len(quarantined),
        conflict_counter=conflict_counter,
        overlap_counter=overlap_counter,
    )

    output.output_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet(output.output_dir / "data" / "train.parquet", reconstructed_rows, config.text_column)
    write_json(output.output_dir / "reports" / "reconstruction_report.json", report)
    write_quarantine(output.output_dir / "reports" / "quarantined_groups.jsonl", quarantined)
    (output.output_dir / "README.md").write_text(dataset_card(report), encoding="utf-8")

    typer.echo(
        "Wrote "
        f"{len(reconstructed_rows)} reconstructed rows and {len(quarantined)} quarantined groups "
        f"to {output.output_dir}."
    )
    if output.no_push or not output.output_repo_id:
        return
    url = publish_output(
        output.output_dir,
        repo_id=output.output_repo_id,
        private=output.private,
        token_env=output.token_env,
    )
    typer.echo(f"Published: {url}")
