from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable, Iterator

import orjson

from wiki_if_builder.config import DEFAULT_DATASET_NAME, DEFAULT_LICENSE
from wiki_if_builder.schemas import (
    AnalystOutput,
    DocumentLabelsDatasetRecord,
    InstructionFollowingDatasetRecord,
    JudgeResult,
    MessageRecord,
)
from wiki_if_builder.utils import JsonlWriter, ensure_dir, iter_jsonl, safe_size


def _as_analyst_output(item: dict[str, Any] | AnalystOutput) -> AnalystOutput:
    return item if isinstance(item, AnalystOutput) else AnalystOutput.model_validate(item)


def iter_analyst_outputs(source: str | Path | Iterable[dict[str, Any] | AnalystOutput]) -> Iterator[AnalystOutput]:
    if isinstance(source, (str, Path)):
        for row in iter_jsonl(source):
            yield AnalystOutput.model_validate(row)
        return
    for item in source:
        yield _as_analyst_output(item)


def load_judge_results(source: str | Path | Iterable[dict[str, Any] | JudgeResult] | None) -> dict[str, JudgeResult]:
    if source is None:
        return {}
    rows: Iterable[dict[str, Any] | JudgeResult]
    rows = iter_jsonl(source) if isinstance(source, (str, Path)) else source
    results: dict[str, JudgeResult] = {}
    for row in rows:
        result = row if isinstance(row, JudgeResult) else JudgeResult.model_validate(row)
        results[result.candidate_id] = result
    return results


def create_train_validation_split(
    records: list[dict[str, Any]],
    validation_ratio: float = 0.02,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not records:
        return [], []
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    validation_size = int(len(shuffled) * validation_ratio)
    if validation_ratio > 0 and validation_size == 0 and len(shuffled) > 1:
        validation_size = 1
    validation = shuffled[:validation_size]
    train = shuffled[validation_size:]
    return train, validation


def _write_dataset_info(dataset_dir: Path, dataset_name: str, row_count: int) -> None:
    info = {
        "dataset_name": dataset_name,
        "license": DEFAULT_LICENSE,
        "splits": {"train": {"name": "train", "num_examples": row_count}},
        "data_files": [{"split": "train", "path": "data.jsonl", "num_examples": row_count}],
        "size_in_bytes": safe_size(dataset_dir / "data.jsonl"),
    }
    (dataset_dir / "dataset_info.json").write_bytes(
        orjson.dumps(info, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
    )


def _labels_readme(dataset_name: str, model_name: str, pipeline_version: str) -> str:
    return f"""---
license: cc-by-sa-3.0
task_categories:
- text-classification
language:
- pt
pretty_name: {dataset_name}
---

# {dataset_name}

Dataset sintético de labels documentais para artigos da Wikipedia em português brasileiro.

## Origem

Os registros derivam de `{DEFAULT_DATASET_NAME}`. Cada linha representa um artigo e preserva `page_id`, `title`, `source_dataset` e a licença herdada `cc-by-sa-3.0`.

## Processo

A pipeline aplica triagem determinística em CPU, remove documentos ruins e usa um modelo local OpenAI-compatible para gerar categorias, subcategorias, tipo documental, affordances de tarefa, elegibilidade para Instruction Following e metadados de qualidade.

## Campos

Inclui status de triagem, contagens de caracteres e palavras, categorias normalizadas, tipo documental, affordances, elegibilidade para IF, bandas de dificuldade, grounding e risco, além de metadados de fallback/truncamento de contexto.

## Modelo

Modelo usado para labeling: `{model_name}`.
Versão da pipeline: `{pipeline_version}`.

## Limitações

Os labels são sintéticos e podem conter erros residuais. Use auditoria humana para aplicações críticas.
"""


def _if_readme(dataset_name: str, model_name: str, judge_model_name: str, pipeline_version: str) -> str:
    return f"""---
license: cc-by-sa-3.0
task_categories:
- text-generation
language:
- pt
pretty_name: {dataset_name}
---

# {dataset_name}

Dataset sintético de Instruction Following em português brasileiro, derivado de artigos da Wikipedia pt-BR.

## Origem

Os exemplos derivam de `{DEFAULT_DATASET_NAME}` e preservam `source_page_id`, `source_title`, `source_dataset` e a licença herdada `cc-by-sa-3.0`.

## Processo

A geração é inspirada em Alpaca e Self-Instruct: cada artigo válido passa por uma chamada de analista documental que produz candidatos de instrução ancorados no texto. Uma etapa opcional de LLM-as-a-Judge avalia grounding, qualidade da instrução, qualidade da resposta, não trivialidade, clareza e validade de schema.

## Campos

Cada linha contém `instruction`, `input`, `output` e `messages`, além de `task_type`, categoria, subcategoria, dificuldade, estilo, metadados do judge e flags de fallback/truncamento. O campo `output` contém apenas a resposta final, sem prompt renderizado, tokens especiais ou conteúdo do judge.

## Modelos

Modelo gerador: `{model_name}`.
Modelo juiz: `{judge_model_name}`.
Versão da pipeline: `{pipeline_version}`.

## Task types

Inclui tarefas como `definition`, `closed_qa`, `summarization`, `information_extraction`, `comparison`, `classification`, `timeline`, `rewrite`, `concept_explanation`, `structured_extraction`, `fact_checking`, `taxonomy`, `short_answer` e `didactic_explanation`.

## Limitações

O dataset é sintético e pode conter erros factuais residuais. Revise amostras antes de uso crítico ou antes de tornar o repositório público.
"""


def export_document_labels_dataset(
    normalized_outputs: str | Path | Iterable[dict[str, Any] | AnalystOutput],
    output_dir: str | Path,
    *,
    dataset_name: str = "wikipedia-pt-br-article-labels-gemma",
    model_name: str = "gemma-local",
    pipeline_version: str = "0.1.0",
) -> Path:
    dataset_dir = ensure_dir(Path(output_dir) / "document_labels")
    data_path = dataset_dir / "data.jsonl"
    count = 0
    seen: set[int] = set()
    with JsonlWriter(data_path, append=False) as writer:
        for output in iter_analyst_outputs(normalized_outputs):
            if output.source.page_id in seen:
                continue
            seen.add(output.source.page_id)
            labels = output.document_labels
            triage = output.triage
            record = DocumentLabelsDatasetRecord(
                id=output.record_id,
                source_dataset=output.source.dataset,
                page_id=output.source.page_id,
                title=output.source.title,
                license=output.source.license,
                triage_status=triage.status,
                quality_flags=triage.quality_flags,
                char_count=triage.char_count,
                word_count=triage.word_count,
                primary_category=labels.primary_category,
                subcategory=labels.subcategory,
                secondary_categories=labels.secondary_categories,
                document_type=labels.document_type,
                task_affordances=labels.task_affordances,
                if_eligibility=labels.if_eligibility,
                difficulty_band=labels.difficulty_band,
                grounding_profile=labels.grounding_profile,
                risk_band=labels.risk_band,
                requires_section_fallback=triage.requires_section_fallback,
                context_truncated=triage.context_truncated,
                used_char_count=triage.used_char_count,
                original_char_count=triage.original_char_count,
                model_name=output.model_name or model_name,
                pipeline_version=output.pipeline_version or pipeline_version,
            )
            writer.write(record.model_dump(mode="json"))
            count += 1

    (dataset_dir / "README.md").write_text(
        _labels_readme(dataset_name, model_name, pipeline_version),
        encoding="utf-8",
    )
    _write_dataset_info(dataset_dir, dataset_name, count)
    return dataset_dir


def _messages_for(instruction: str, input_text: str, output_text: str) -> list[MessageRecord]:
    user_content = instruction.strip()
    if input_text.strip():
        user_content = f"{user_content}\n\nContexto:\n{input_text.strip()}"
    return [
        MessageRecord(role="user", content=user_content),
        MessageRecord(role="assistant", content=output_text.strip()),
    ]


def export_instruction_following_dataset(
    normalized_outputs: str | Path | Iterable[dict[str, Any] | AnalystOutput],
    judge_results: str | Path | Iterable[dict[str, Any] | JudgeResult] | None,
    output_dir: str | Path,
    *,
    include_review: bool = False,
    dataset_name: str = "wikipedia-pt-br-instructions-gemma",
    model_name: str = "gemma-local",
    judge_model_name: str = "gemma-local",
    pipeline_version: str = "0.1.0",
) -> Path:
    dataset_dir = ensure_dir(Path(output_dir) / "instruction_following")
    data_path = dataset_dir / "data.jsonl"
    judges = load_judge_results(judge_results)
    count = 0
    with JsonlWriter(data_path, append=False) as writer:
        for output in iter_analyst_outputs(normalized_outputs):
            labels = output.document_labels
            for candidate in output.if_candidates:
                judge = judges.get(candidate.candidate_id)
                if judge is None:
                    continue
                verdict = judge.judge.verdict
                if verdict == "reject" or (verdict == "review" and not include_review):
                    continue
                scores = judge.judge.scores
                messages = _messages_for(
                    candidate.instruction,
                    candidate.input,
                    candidate.completion,
                )
                record = InstructionFollowingDatasetRecord(
                    id=candidate.candidate_id,
                    source_dataset=output.source.dataset,
                    source_page_id=output.source.page_id,
                    source_title=output.source.title,
                    license=output.source.license,
                    instruction=candidate.instruction,
                    input=candidate.input,
                    output=candidate.completion,
                    messages=messages,
                    task_type=candidate.task_type,
                    primary_category=labels.primary_category,
                    subcategory=labels.subcategory,
                    difficulty=candidate.difficulty,
                    style=candidate.style,
                    judge_verdict=verdict,
                    judge_score=judge.judge.overall_score,
                    judge_grounding=scores.grounding,
                    judge_instruction_quality=scores.instruction_quality,
                    judge_completion_quality=scores.completion_quality,
                    judge_non_triviality=scores.non_triviality,
                    requires_section_fallback=output.triage.requires_section_fallback,
                    context_truncated=output.triage.context_truncated,
                    model_name=output.model_name or model_name,
                    judge_model_name=judge.judge_model_name or judge_model_name,
                    pipeline_version=output.pipeline_version or pipeline_version,
                )
                writer.write(record.model_dump(mode="json"))
                count += 1

    (dataset_dir / "README.md").write_text(
        _if_readme(dataset_name, model_name, judge_model_name, pipeline_version),
        encoding="utf-8",
    )
    _write_dataset_info(dataset_dir, dataset_name, count)
    return dataset_dir


def export_all_from_intermediate(
    output_dir: str | Path,
    *,
    include_review: bool = False,
    model_name: str = "gemma-local",
    judge_model_name: str = "gemma-local",
    pipeline_version: str = "0.1.0",
) -> tuple[Path, Path]:
    intermediate = Path(output_dir) / "intermediate"
    normalized_path = intermediate / "normalized_outputs.jsonl"
    judge_path = intermediate / "judge_results.jsonl"
    labels_dir = export_document_labels_dataset(
        normalized_path,
        output_dir,
        model_name=model_name,
        pipeline_version=pipeline_version,
    )
    if_dir = export_instruction_following_dataset(
        normalized_path,
        judge_path,
        output_dir,
        include_review=include_review,
        model_name=model_name,
        judge_model_name=judge_model_name,
        pipeline_version=pipeline_version,
    )
    return labels_dir, if_dir

