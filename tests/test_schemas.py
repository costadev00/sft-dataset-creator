from wiki_if_builder.schemas import AnalystOutput


def test_schemas_validate_correct_analyst_output():
    payload = {
        "record_id": "wiki-ptbr-12345",
        "source": {
            "dataset": "costadev00/wikipedia-pt-br-extract",
            "page_id": 12345,
            "title": "Astronomia",
            "license": "cc-by-sa-3.0",
        },
        "triage": {
            "status": "valid_article",
            "quality_flags": [],
            "char_count": 1000,
            "word_count": 150,
            "unique_word_count": 80,
            "alpha_ratio": 0.8,
            "symbolic_ratio": 0.05,
            "reason": "ok",
            "requires_section_fallback": False,
            "context_truncated": False,
            "used_char_count": 1000,
            "original_char_count": 1000,
        },
        "document_labels": {
            "primary_category": "ciencia",
            "subcategory": "astronomia",
            "secondary_categories": ["fisica"],
            "document_type": "encyclopedic_article",
            "task_affordances": ["definition", "closed_qa"],
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
                "instruction": "Explique o que é astronomia.",
                "input": "Contexto mínimo.",
                "completion": "Astronomia é uma ciência natural.",
                "style": "didatico",
                "difficulty": "easy",
                "source_refs": [{"section_id": 0, "char_start": 0, "char_end": 500}],
            }
        ],
    }
    output = AnalystOutput.model_validate(payload)
    assert output.source.page_id == 12345
    assert output.if_candidates[0].candidate_id == "wiki-ptbr-12345-c1"

