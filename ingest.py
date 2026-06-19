"""
ingest.py — jednorazowy skrypt budujący bazę wektorową FAISS.

Uruchom przed pierwszym startem aplikacji:
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

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
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


# ── Chunking z zachowaniem artykułów ─────────────────────────────────────────

def split_by_articles(text: str, source: str) -> list[dict]:
    article_pattern = re.compile(r'(Art\.\s*\d+[a-z]?\.)', re.IGNORECASE)
    parts = article_pattern.split(text)

    chunks = []
    current_article = "brak numeru"
    current_text = ""

    for part in parts:
        if article_pattern.match(part):
            current_article = part.strip()
        else:
            current_text += part

        while len(current_text) > CHUNK_SIZE:
            chunk = current_text[:CHUNK_SIZE]
            chunks.append({
                "text": chunk.strip(),
                "source": source,
                "article": current_article,
            })
            current_text = current_text[CHUNK_SIZE - CHUNK_OVERLAP:]

    if current_text.strip():
        chunks.append({
            "text": current_text.strip(),
            "source": source,
            "article": current_article,
        })

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
        print(f"   Wyciągnięto {len(text):,} znaków")

        chunks = split_by_articles(text, source_name)
        print(f"   Powstało {len(chunks)} chunków")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("\n❌ Brak danych. Sprawdź czy pliki PDF są w folderze data/")
        return

    print(f"\n🔢 Generuję embeddingi dla {len(all_chunks)} chunków...")
    model = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings).astype("float32")

    print("\n💾 Zapisuję indeks FAISS...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Gotowe! Zaindeksowano {len(all_chunks)} chunków.")
    print(f"   {FAISS_INDEX_PATH}")
    print(f"   {METADATA_PATH}")


if __name__ == "__main__":
    build_vectorstore()
