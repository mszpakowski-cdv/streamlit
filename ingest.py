"""
ingest.py — jednorazowy skrypt budujący bazę wektorową FAISS.

Uruchom przed pierwszym startem aplikacji (i po każdej zmianie chunkowania):
    python ingest.py

Wymagania:
    pip install -r requirements.txt
"""

import os
import re
import sys
import json
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

# Wymuś UTF-8 na stdout — inaczej polskie konsole (cp1250) wywalają się na emoji.
sys.stdout.reconfigure(encoding="utf-8")

# ── Konfiguracja ──────────────────────────────────────────────────────────────

PDF_FILES = {
    "kpw": "data/kodeks.pdf",  # Kodeks POSTĘPOWANIA w sprawach o wykroczenia
    "prawo_o_ruchu_drogowym": "data/ruch_drogowy.pdf",
}

# Taryfikator mandatów — dane już ustrukturyzowane (jeden rekord = jedno wykroczenie)
TARYFIKATOR_JSON = "data/taryfikator.json"

# Czytelne nazwy ustaw — używane do wzbogacenia embeddingu
SOURCE_LABELS = {
    "kpw": "Kodeks postępowania w sprawach o wykroczenia",
    "prawo_o_ruchu_drogowym": "Prawo o ruchu drogowym",
    "taryfikator": "Taryfikator mandatów",
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
    """Usuwa stopki Kancelarii Sejmu, nagłówki sekcji i nadmiarowe spacje."""
    # Usuń powtarzające się stopki typu "©Kancelaria Sejmu  s. 12/61"
    text = re.sub(r'©Kancelaria Sejmu\s+s\.\s*\d+/\d+', ' ', text)
    # Usuń daty stopki typu "2023-10-17"
    text = re.sub(r'\b20\d{2}-\d{2}-\d{2}\b', ' ', text)
    # Usuń nagłówki sekcji ("Oddział 7 ...", "Rozdział I ...", "DZIAŁ II ...") —
    # to nawigacja, nie treść przepisu; w chunkach tylko zaśmiecała kontekst.
    text = re.sub(
        r'\b(?:DZIAŁ|Dział|Rozdział|Oddział)\s+[IVXLCDM0-9]+[a-z]?\b[^\n]*',
        ' ',
        text,
    )
    # Zredukuj wielokrotne spacje/odstępy do pojedynczej spacji
    text = re.sub(r'[ \t]+', ' ', text)
    return text


# ── Chunking z zachowaniem artykułów ─────────────────────────────────────────

def split_by_articles(text: str, source: str) -> list[dict]:
    article_pattern = re.compile(r'(Art\.\s*\d+[a-z]?\.)', re.IGNORECASE)
    parts = article_pattern.split(text)
    label = SOURCE_LABELS.get(source, source)

    chunks = []
    current_article = "brak numeru"
    buffer = ""

    def flush():
        """Zapisuje cały tekst bieżącego artykułu (cięty na <= CHUNK_SIZE)."""
        nonlocal buffer
        article_text = buffer.strip()
        buffer = ""
        if not article_text:
            return

        start = 0
        n = len(article_text)
        while start < n:
            end = start + CHUNK_SIZE
            if end < n:
                cut = article_text.rfind(" ", start, end)
                if cut <= start:
                    cut = end
            else:
                cut = n
            piece = article_text[start:cut].strip()
            if piece:
                chunks.append({
                    "text": piece,
                    # Tekst do embeddingu wzbogacony o ustawę i artykuł — poprawia RAG.
                    "embed_text": f"{label}, {current_article}\n{piece}",
                    "source": source,
                    "article": current_article,
                })
            if cut >= n:
                break
            start = max(cut - CHUNK_OVERLAP, start + 1)

    for part in parts:
        if article_pattern.match(part):
            # Nowy artykuł: najpierw zapisz poprzedni pod JEGO numerem, dopiero
            # potem przełącz etykietę (inaczej ogon poprzedniego artykułu trafiał
            # do następnego — tak Art. 24 lądował pod "Art. 25").
            flush()
            current_article = part.strip()
        else:
            buffer += part

    flush()
    return chunks


# ── Taryfikator mandatów (dane tabelaryczne — jeden chunk = jedno wykroczenie) ──

def build_taryfikator_chunks() -> list[dict]:
    if not os.path.exists(TARYFIKATOR_JSON):
        print(f"[POMINIĘTO] Nie znaleziono pliku: {TARYFIKATOR_JSON}")
        return []
    rows = json.load(open(TARYFIKATOR_JSON, encoding="utf-8"))
    label = SOURCE_LABELS["taryfikator"]
    chunks = []
    for r in rows:
        # Tekst wyświetlany — pełna pozycja taryfikatora.
        text = (f"{r['opis']} — grzywna {r['grzywna']} zł "
                f"(kwalifikacja: {r['kw_art']} k.w.; naruszony przepis: {r['pord_ref']}).")
        # Tekst do embeddingu — wzbogacony o sekcję i powiązany przepis ruchu drogowego,
        # żeby krótkie pozycje liczbowe dobrze się wyszukiwały dla opisowych zapytań.
        embed_text = (f"{label}, {r['sekcja']}. {r['opis']}. "
                      f"Naruszony przepis ruchu drogowego: {r['pord_ref']}. "
                      f"Kwalifikacja prawna: {r['kw_art']} Kodeksu wykroczeń. "
                      f"Grzywna (mandat): {r['grzywna']} zł.")
        chunks.append({
            "text": text,
            "embed_text": embed_text,
            "source": "taryfikator",
            "article": f"{r['kw_art']} (taryfikator lp. {r['lp']})",
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
        text = clean_text(text)
        print(f"   Wyciągnięto {len(text):,} znaków")

        chunks = split_by_articles(text, source_name)
        print(f"   Powstało {len(chunks)} chunków")
        all_chunks.extend(chunks)

    taryf_chunks = build_taryfikator_chunks()
    if taryf_chunks:
        print(f"\n💰 Taryfikator mandatów: {len(taryf_chunks)} pozycji")
        all_chunks.extend(taryf_chunks)

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