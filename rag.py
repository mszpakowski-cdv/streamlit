"""
rag.py — wyszukiwanie semantyczne po zaindeksowanych dokumentach prawnych.
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-m3"
FAISS_INDEX_PATH = "vectorstore/index.faiss"
METADATA_PATH = "vectorstore/metadata.json"

_model = None
_index = None
_metadata = None


def _load_resources():
    global _model, _index, _metadata
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    if _index is None:
        _index = faiss.read_index(FAISS_INDEX_PATH)
    if _metadata is None:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            _metadata = json.load(f)


def retrieve(query: str, k: int = 5) -> list[dict]:
    """Zwraca k najbardziej pasujących chunków do zapytania."""
    _load_resources()
    query_embedding = _model.encode(
        [query], normalize_embeddings=True
    ).astype("float32")
    distances, indices = _index.search(query_embedding, k)
    results = []
    for idx in indices[0]:
        if idx != -1:
            results.append(_metadata[idx])
    return results


def format_context(chunks: list[dict]) -> str:
    """Formatuje chunki do czytelnego kontekstu dla LLM."""
    parts = []
    for chunk in chunks:
        source_label = {
            "kodeks_wykroczen": "Kodeks postępowania w sprawach o wykroczenia",
            "prawo_o_ruchu_drogowym": "Prawo o ruchu drogowym",
        }.get(chunk["source"], chunk["source"])
        parts.append(f"[{source_label} — {chunk['article']}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)