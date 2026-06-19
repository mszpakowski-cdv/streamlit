"""
ingest.py — jednorazowy skrypt budujący bazę wektorową FAISS.

Uruchom przed pierwszym startem aplikacji (i po każdej zmianie chunkowania):
    python ingest.py

Wymagania:
    pip install -r requirements.txt
"""

import os
import re
import json
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

# ── Konfiguracja ──────────────────────────────────────────────────────────────

PDF_FILES = {
    "kodeks_wykroczen": "data/kodeks.pdf",
    "prawo_o_ruchu_drogowym": "data/ruch_drogowy.pdf",
}

# Czytelne nazwy ustaw — używane do wzbogacenia embeddingu
SOURCE_LABELS = {
    "kodeks_wykroczen": "Kodeks postępowania w sprawach o wykroczenia",
    "prawo_o_ruchu_drogowym": "Prawo o ruchu drogowym",
}

EMBED_MODEL = "BAAI/bge-m3"
OUTPUT_DIR = "vectorstore"
FAISS_INDEX_PATH = os.path.join(OUTPUT_DIR, "index.faiss")
METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.json")
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


# ── Ekstrakcja tekstu ─────────────────────────────────────────────────────────

def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


# ── Czyszczenie tekstu ────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Usuwa stopki Kancelarii Sejmu i nadmiarowe spacje z ekstrakcji PDF."""
    # Usuń powtarzające się stopki typu "©Kancelaria Sejmu  s. 12/61"
    text = re.sub(r'©Kancelaria Sejmu\s+s\.\s*\d+/\d+', ' ', text)
    # Usuń daty stopki typu "2023-10-17"
    text = re.sub(r'\b20\d{2}-\d{2}-\d{2}\b', ' ', text)
    # Zredukuj wielokrotne spacje/odstępy do pojedynczej spacji
    text = re.sub(r'[ \t]+', ' ', text)
    return text


# ── Chunking z zachowaniem artykułów ─────────────────────────────────────────

def split_by_articles(text: str, source: str) -> list[dict]:
    article_pattern = re.compile(r'(Art\.\s*\d+[a-z]?\.)', re.IGNORECASE)
    parts = article_pattern.split(text)

    chunks = []
    current_article = "brak numeru"
    current_text = ""

    def emit(chunk_text: str):
        chunk_text = chunk_text.strip()
        if not chunk_text:
            return
        label = SOURCE_LABELS.get(source, source)
        # Tekst do embeddingu wzbogacony o ustawę i artykuł — poprawia trafność RAG.
        # Surowy tekst (do wyświetlania) trzymamy osobno w "text".
        embed_text = f"{label}, {current_article}\n{chunk_text}"
        chunks.append({
            "text": chunk_text,
            "embed_text": embed_text,
            "source": source,
            "article": current_article,
        })

    for part in parts:
        if article_pattern.match(part):
            current_article = part.strip()
        else:
            current_text += part

        while len(current_text) > CHUNK_SIZE:
            cut = current_text.rfind(" ", 0, CHUNK_SIZE)
            if cut == -1:
                cut = CHUNK_SIZE
            emit(current_text[:cut])
            overlap_start = cut - CHUNK_OVERLAP
            space = current_text.rfind(" ", 0, overlap_start)
            current_text = current_text[(space + 1) if space != -1 else overlap_start:]

    if current_text.strip():
        emit(current_text)

    return chunks


# ── Główna funkcja ─────────────────────────────────────────────────────────────

def build_vectorstore():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_chunks = []

    for source_name, pdf_path in PDF_FILES.items():
        if not os.path.exists(pdf_path):
            print(f"[POMINIĘTO] Nie znaleziono pliku: {pdf_path}")
            continue

        print(f"\n📄 Przetwarzam: {pdf_path}")
        text = extract_text_from_pdf(pdf_path)
        text = clean_text(text)
        print(f"   Wyciągnięto {len(text):,} znaków")

        chunks = split_by_articles(text, source_name)
        print(f"   Powstało {len(chunks)} chunków")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("\n❌ Brak danych. Sprawdź czy pliki PDF są w folderze data/")
        return

    print(f"\n🔢 Generuję embeddingi dla {len(all_chunks)} chunków...")
    model = SentenceTransformer(EMBED_MODEL)
    # Embedujemy wzbogacony tekst (z nazwą ustawy i artykułem)
    texts = [c["embed_text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=16, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")

    print("\n💾 Zapisuję indeks FAISS...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)

    # W metadanych nie zapisujemy już embed_text (niepotrzebny przy wyświetlaniu)
    metadata = [
        {"text": c["text"], "source": c["source"], "article": c["article"]}
        for c in all_chunks
    ]
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Gotowe! Zaindeksowano {len(all_chunks)} chunków.")
    print(f"   {FAISS_INDEX_PATH}")
    print(f"   {METADATA_PATH}")


if __name__ == "__main__":
    build_vectorstore()