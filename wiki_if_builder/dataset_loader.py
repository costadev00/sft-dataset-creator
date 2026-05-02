from __future__ import annotations

from itertools import islice
from typing import Any, Iterable, Iterator

from datasets import load_dataset

from wiki_if_builder.config import DEFAULT_DATASET_NAME
from wiki_if_builder.utils import normalize_page_id


EXPECTED_FIELDS = ("page_id", "title", "text", "ns", "section_texts")


def normalize_article(record: dict[str, Any]) -> dict[str, Any]:
    section_texts = record.get("section_texts") or []
    if isinstance(section_texts, str):
        section_texts = [section_texts]
    elif not isinstance(section_texts, list):
        section_texts = list(section_texts) if section_texts else []

    return {
        "page_id": normalize_page_id(record.get("page_id")),
        "title": str(record.get("title") or ""),
        "text": str(record.get("text") or ""),
        "ns": record.get("ns", 0),
        "section_texts": section_texts,
    }


def load_wikipedia_dataset(
    dataset_name: str = DEFAULT_DATASET_NAME,
    split: str = "train",
    cache_dir: str | None = None,
    streaming: bool = True,
):
    return load_dataset(dataset_name, split=split, cache_dir=cache_dir, streaming=streaming)


def iter_articles(dataset: Iterable[dict[str, Any]], max_articles: int | None = None) -> Iterator[dict[str, Any]]:
    source = dataset if max_articles is None else islice(dataset, max_articles)
    for record in source:
        yield normalize_article(record)


def synthetic_dry_run_articles(limit: int | None = None) -> Iterator[dict[str, Any]]:
    articles = [
        {
            "page_id": 1001,
            "title": "Astronomia",
            "ns": 0,
            "section_texts": [
                "Astronomia é a ciência natural que estuda corpos celestes, fenômenos fora da atmosfera terrestre e a evolução do universo observável.",
                "A astronomia moderna combina observações, modelos matemáticos, instrumentos ópticos e análise de dados para investigar planetas, estrelas, galáxias e cosmologia.",
                "No Brasil, a divulgação científica em astronomia aparece em planetários, universidades e projetos de observação do céu.",
            ],
            "text": (
                "Astronomia é a ciência natural que estuda corpos celestes, fenômenos fora da atmosfera terrestre "
                "e a evolução do universo observável. A disciplina investiga planetas, estrelas, galáxias, nebulosas, "
                "cometas, asteroides e a radiação emitida por esses objetos. A astronomia moderna combina observações "
                "sistemáticas, modelos matemáticos, instrumentos ópticos, radiotelescópios e análise computacional de dados. "
                "Historicamente, a observação do céu ajudou sociedades humanas a organizar calendários, navegação e práticas "
                "culturais. Hoje, a área dialoga com física, matemática, engenharia, computação e educação científica."
            ),
        },
        {
            "page_id": 1002,
            "title": "Machado de Assis",
            "ns": 0,
            "section_texts": [
                "Joaquim Maria Machado de Assis foi um escritor brasileiro, considerado um dos nomes centrais da literatura em língua portuguesa.",
                "Sua obra inclui romances, contos, crônicas, poemas e peças teatrais, com destaque para Memórias Póstumas de Brás Cubas e Dom Casmurro.",
            ],
            "text": (
                "Joaquim Maria Machado de Assis foi um escritor brasileiro nascido no Rio de Janeiro. É amplamente reconhecido "
                "como um dos principais autores da literatura em língua portuguesa. Sua produção inclui romances, contos, crônicas, "
                "poemas e peças teatrais. Entre suas obras mais conhecidas estão Memórias Póstumas de Brás Cubas, Quincas Borba "
                "e Dom Casmurro. Machado participou da fundação da Academia Brasileira de Letras e tornou-se referência por sua "
                "ironia, experimentação narrativa e observação crítica da sociedade brasileira do século XIX."
            ),
        },
        {
            "page_id": 1003,
            "title": "Bacia Amazônica",
            "ns": 0,
            "section_texts": [
                "A Bacia Amazônica é uma extensa bacia hidrográfica da América do Sul, associada ao rio Amazonas e seus afluentes.",
                "Ela atravessa vários países e tem grande relevância ecológica, climática e social.",
            ],
            "text": (
                "A Bacia Amazônica é uma das maiores bacias hidrográficas do mundo e está associada ao rio Amazonas e a uma rede "
                "complexa de afluentes. Ela se estende por diversos países da América do Sul, incluindo Brasil, Peru, Colômbia e Bolívia. "
                "A região possui enorme relevância ecológica por abrigar florestas tropicais, grande biodiversidade e ciclos hidrológicos "
                "que influenciam o clima regional. A bacia também é importante para populações ribeirinhas, povos indígenas, transporte, "
                "pesca, pesquisa científica e políticas de conservação ambiental."
            ),
        },
    ]
    count = len(articles) if limit is None else min(limit, len(articles))
    for article in articles[:count]:
        yield normalize_article(article)

