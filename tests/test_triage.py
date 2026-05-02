from wiki_if_builder.triage import is_valid_for_llm, triage_article


def _article(text: str, ns: int = 0, title: str = "Artigo Teste"):
    return {"page_id": 1, "title": title, "text": text, "ns": ns, "section_texts": [text]}


def test_triage_rejects_symbolic_garbage():
    result = triage_article(_article("####"))
    assert result.status == "empty_or_symbolic"
    assert "empty_or_too_short" in result.quality_flags


def test_triage_rejects_empty_text():
    result = triage_article(_article(""))
    assert result.status == "empty_or_symbolic"


def test_triage_rejects_non_main_namespace():
    text = "Texto enciclopédico " * 100
    result = triage_article(_article(text, ns=14, title="Categoria:Teste"))
    assert result.status == "non_main_namespace"


def test_triage_accepts_simple_encyclopedic_article():
    text = (
        "A astronomia é uma ciência natural dedicada ao estudo de corpos celestes, fenômenos físicos, "
        "observações sistemáticas, instrumentos ópticos, calendários, navegação, modelos matemáticos, "
        "estrelas, planetas, galáxias, nebulosas, cometas, asteroides, radiação, cosmologia, educação, "
        "história, cultura, pesquisa, universidades, telescópios, dados, espectroscopia, órbitas, gravidade, "
        "luz, espaço, tempo, matéria, energia, atmosfera, divulgação, planetários, satélites e missões espaciais. "
    ) * 3
    result = triage_article(_article(text, title="Astronomia"))
    assert is_valid_for_llm(result)

