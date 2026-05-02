from wiki_if_builder.normalizer import normalize_analyst_output, slugify_pt
from wiki_if_builder.schemas import AnalystOutput


def _sample_output():
    return AnalystOutput.model_validate(
        {
            "record_id": "wiki-ptbr-7",
            "source": {
                "dataset": "costadev00/wikipedia-pt-br-extract",
                "page_id": 7,
                "title": "Computação",
                "license": "cc-by-sa-3.0",
            },
            "triage": {
                "status": "valid_article",
                "quality_flags": [],
                "char_count": 1000,
                "word_count": 120,
                "unique_word_count": 70,
                "alpha_ratio": 0.8,
                "symbolic_ratio": 0.03,
                "reason": "ok",
                "used_char_count": 1000,
                "original_char_count": 1000,
            },
            "document_labels": {
                "primary_category": "Ciência da Computação",
                "subcategory": "História da Computação",
                "secondary_categories": ["Tecnologia da Informação"],
                "document_type": "Technical Article",
                "task_affordances": ["Closed QA"],
                "if_eligibility": True,
                "difficulty_band": "Médio",
                "grounding_profile": "Strong",
                "risk_band": "Baixo",
            },
            "evidence_refs": [{"section_id": 0, "char_start": 0, "char_end": 500}],
            "if_candidates": [
                {
                    "candidate_id": "wiki-ptbr-7-c1",
                    "task_type": "Closed QA",
                    "instruction": "O que é computação?",
                    "input": "Contexto",
                    "completion": "Computação é o estudo de processos de informação.",
                    "style": "Didático",
                    "difficulty": "Easy",
                    "source_refs": [],
                },
                {
                    "candidate_id": "wiki-ptbr-7-c2",
                    "task_type": "closed_qa",
                    "instruction": "",
                    "input": "Contexto",
                    "completion": "",
                    "style": "didatico",
                    "difficulty": "easy",
                    "source_refs": [],
                },
            ],
        }
    )


def test_normalizer_converts_categories_to_snake_case():
    normalized = normalize_analyst_output(_sample_output())
    assert normalized.document_labels.primary_category == "ciencia_da_computacao"
    assert normalized.document_labels.subcategory == "historia_da_computacao"


def test_normalizer_removes_empty_candidates():
    normalized = normalize_analyst_output(_sample_output())
    assert len(normalized.if_candidates) == 1
    assert normalized.if_candidates[0].source_refs


def test_slugify_pt_removes_accents():
    assert slugify_pt("Saúde Pública") == "saude_publica"

