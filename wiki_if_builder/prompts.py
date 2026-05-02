from __future__ import annotations

import json
from typing import Any


DOCUMENT_ANALYST_SYSTEM_PROMPT = """Você é um analista documental para artigos da Wikipedia em português brasileiro.
Retorne somente JSON válido conforme o schema AnalystOutput. Não use markdown, não adicione explicações fora do JSON e não inclua tokens especiais de chat.

Regras:
1. Não invente informações que não estejam apoiadas no artigo.
2. Gere labels documentais, affordances de tarefa e de 2 a 4 candidatos de Instruction Following.
3. Varie tipos de tarefa e evite perguntas triviais.
4. Evite completions longas demais.
5. Não inclua chain of thought, raciocínio oculto ou conteúdo especulativo.
6. Mantenha português brasileiro.
7. Use source_refs com section_id, char_start e char_end para ancorar cada candidato.
8. Não salve prompt renderizado com template específico de modelo.

Task types permitidos:
definition, closed_qa, summarization, information_extraction, comparison, classification, timeline, rewrite, concept_explanation, structured_extraction, fact_checking, taxonomy, short_answer, didactic_explanation.

primary_category sugeridas:
ciencia, tecnologia, historia, geografia, biografia, cultura, artes, literatura, politica, economia, direito, saude, educacao, esportes, religiao, filosofia, sociedade, linguistica, matematica, engenharia, meio_ambiente, entretenimento, outros.

document_type permitido:
encyclopedic_article, biography, historical_article, geographical_article, scientific_article, technical_article, cultural_article, list_like, stub, redirect, disambiguation, malformed, unknown.
"""


JUDGE_SYSTEM_PROMPT = """Você é um avaliador pointwise de exemplos Instruction Following derivados da Wikipedia.
Retorne somente JSON válido. Não use markdown e não gere nova completion.

Critérios:
1. Reject se a completion não for suportada pelas evidências.
2. Reject se a instruction for vaga demais.
3. Reject se houver alucinação factual.
4. Reject se a resposta for excessivamente verbosa sem ganho.
5. Review se o candidato for bom, mas depender de revisão humana.
6. Accept se for factual, claro, útil e bem ancorado no artigo.
7. Penalize respostas genéricas, tarefas triviais demais e exemplos que dependem de contexto ausente.
"""


ANALYST_SCHEMA_HINT = {
    "record_id": "wiki-ptbr-<page_id>",
    "source": {
        "dataset": "costadev00/wikipedia-pt-br-extract",
        "page_id": 12345,
        "title": "Título",
        "license": "cc-by-sa-3.0",
    },
    "triage": {
        "status": "valid_article",
        "quality_flags": [],
        "requires_section_fallback": False,
        "context_truncated": False,
        "used_char_count": 100000,
        "original_char_count": 100000,
    },
    "document_labels": {
        "primary_category": "ciencia",
        "subcategory": "astronomia",
        "secondary_categories": ["fisica"],
        "document_type": "encyclopedic_article",
        "task_affordances": ["definition", "closed_qa", "summarization"],
        "if_eligibility": True,
        "difficulty_band": "medium",
        "grounding_profile": "strong",
        "risk_band": "low",
    },
    "evidence_refs": [{"section_id": 0, "char_start": 0, "char_end": 500}],
    "if_candidates": [
        {
            "candidate_id": "wiki-ptbr-12345-c1",
            "task_type": "closed_qa",
            "instruction": "Explique o tema do artigo.",
            "input": "Contexto mínimo necessário.",
            "completion": "Resposta factual apoiada no artigo.",
            "style": "didatico",
            "difficulty": "easy",
            "source_refs": [{"section_id": 0, "char_start": 0, "char_end": 500}],
        }
    ],
}


def render_analyst_user_prompt(
    *,
    record_id: str,
    page_id: int,
    title: str,
    dataset_name: str,
    license_name: str,
    triage: dict[str, Any],
    context_text: str,
    candidates_per_article: int,
) -> str:
    return (
        "Analise o artigo abaixo e retorne somente JSON válido no schema AnalystOutput.\n"
        f"record_id: {record_id}\n"
        f"dataset: {dataset_name}\n"
        f"page_id: {page_id}\n"
        f"title: {title}\n"
        f"license: {license_name}\n"
        f"candidates_per_article desejado: {candidates_per_article}\n\n"
        "Triage determinística já aplicada:\n"
        f"{json.dumps(triage, ensure_ascii=False, indent=2)}\n\n"
        "Schema de referência:\n"
        f"{json.dumps(ANALYST_SCHEMA_HINT, ensure_ascii=False, indent=2)}\n\n"
        "Artigo com seções e offsets aproximados:\n"
        f"{context_text}"
    )


def render_judge_user_prompt(
    *,
    title: str,
    page_id: int,
    candidate: dict[str, Any],
    evidence_snippets: list[str],
) -> str:
    payload = {
        "title": title,
        "page_id": page_id,
        "candidate": candidate,
        "evidence_snippets": evidence_snippets,
        "expected_output_schema": {
            "candidate_id": candidate.get("candidate_id", ""),
            "judge": {
                "verdict": "accept|reject|review",
                "overall_score": 4.5,
                "scores": {
                    "grounding": 5,
                    "instruction_quality": 4,
                    "completion_quality": 5,
                    "non_triviality": 4,
                    "style_clarity": 5,
                    "schema_validity": 5,
                },
                "issues": [],
                "needs_human_review": False,
            },
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)

