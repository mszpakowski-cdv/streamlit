"""
rag.py — wyszukiwanie semantyczne po zaindeksowanych dokumentach prawnych.

Dwustopniowy retrieval:
  1. FAISS (dense, bge-m3) — szybko wyławia kandydatów.
  2. Reranker (cross-encoder, bge-reranker-v2-m3) — przelicza trafność każdej
     pary (zapytanie, fragment) i odrzuca fragmenty poniżej progu.
Cross-encoder jest dużo precyzyjniejszy niż samo podobieństwo wektorowe, więc
do modelu trafia mniej szumu (a mniej szumu = mniej zmyślonych naruszeń).
"""

import json
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
FAISS_INDEX_PATH = "vectorstore/index.faiss"
METADATA_PATH = "vectorstore/metadata.json"

# Ilu kandydatów pobieramy wektorowo, zanim przepuścimy ich przez reranker.
# 10 zamiast 20: reranker (cross-encoder na CPU) jest ~2x szybszy, a trafność top-k
# nie spada (mniej szumu dla rerankera → trafny artykuł często wskakuje wyżej).
CANDIDATE_K = 10
# Próg trafności reranku. bge-reranker-v2-m3 zwraca logity: ~0 to granica
# (sigmoid 0.5). Fragmenty poniżej progu odrzucamy jako nietrafne.
RERANK_THRESHOLD = 0.0
# Ile fragmentów minimum zwrócić, nawet jeśli próg odrzuciłby wszystko —
# żeby model dostał choć najlepszy kontekst (i tak ma prawo orzec brak naruszenia).
MIN_KEEP = 3

_embed_model = None
_reranker = None
_index = None
_metadata = None


def _load_resources():
    global _embed_model, _reranker, _index, _metadata
    # Domyślnie CPU: bge-m3 + reranker nie zabierają wtedy VRAM Bielikowi (karta 6 GB).
    # Eval może wymusić GPU przez RAG_DEVICE=cuda (dwufazowy przebieg: najpierw cały
    # retrieval na GPU bez Bielika, potem inferencja Bielika) — wtedy retrieval jest
    # ~20x szybszy i znika przestój między przypadkami.
    import os
    device = os.environ.get("RAG_DEVICE", "cpu")
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                device = "cpu"  # torch bez CUDA — i tak nie ma GPU dla embeddera
        except Exception:
            device = "cpu"
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL, device=device)
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL, device=device)
    if _index is None:
        _index = faiss.read_index(FAISS_INDEX_PATH)
    if _metadata is None:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            _metadata = json.load(f)


def free_resources():
    """Zwalnia embedder + reranker z pamięci (GPU) — używane przez eval między fazami."""
    global _embed_model, _reranker
    _embed_model = None
    _reranker = None
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def retrieve(
    query: str,
    k: int = 5,
    candidate_k: int = CANDIDATE_K,
    threshold: float = RERANK_THRESHOLD,
    min_keep: int = MIN_KEEP,
) -> list[dict]:
    """Zwraca do k najtrafniejszych fragmentów (każdy z polem 'score' z reranku).

    1. FAISS zwraca candidate_k kandydatów.
    2. Reranker ocenia każdą parę (zapytanie, fragment).
    3. Zostają fragmenty >= threshold (maks. k); jeśli jest ich mniej niż
       min_keep — dobieramy najlepsze niezależnie od progu.
    """
    _load_resources()

    query_embedding = _embed_model.encode(
        [query], normalize_embeddings=True
    ).astype("float32")
    _, indices = _index.search(query_embedding, candidate_k)

    candidates = [_metadata[idx] for idx in indices[0] if idx != -1]
    if not candidates:
        return []

    scores = _reranker.predict([(query, c["text"]) for c in candidates])
    ranked = sorted(
        ({**c, "score": float(s)} for c, s in zip(candidates, scores)),
        key=lambda c: c["score"],
        reverse=True,
    )

    kept = [c for c in ranked if c["score"] >= threshold][:k]
    if len(kept) < min_keep:
        kept = ranked[:min_keep]
    return kept


def format_context(chunks: list[dict]) -> str:
    """Formatuje chunki do czytelnego kontekstu dla LLM."""
    parts = []
    for chunk in chunks:
        source_label = {
            "kpw": "Kodeks postępowania w sprawach o wykroczenia",
            "prawo_o_ruchu_drogowym": "Prawo o ruchu drogowym",
            "taryfikator": "Taryfikator mandatów",
        }.get(chunk["source"], chunk["source"])
        parts.append(f"[{source_label} — {chunk['article']}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)
