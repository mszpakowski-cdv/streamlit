# 🚔 Agent Kodeksu Drogowego

Aplikacja Streamlit wspierająca policjantów drogówki podczas interwencji.
Policjant opisuje sytuację, agent wskazuje naruszone przepisy i grożące kary.

## Stack
- **Model:** Bielik (`speakleash/Bielik-11B-v2.3-Instruct`) przez HuggingFace Inference API
- **RAG:** FAISS + sentence-transformers (embeddingi po polsku)
- **Guardrails:** filtrowanie zapytań niezwiązanych z ruchem drogowym
- **UI:** Streamlit

## Struktura projektu
```
traffic-agent/
├── app.py              # główna aplikacja
├── rag.py              # wyszukiwanie semantyczne
├── guardrails.py       # filtrowanie zapytań
├── ingest.py           # budowanie bazy wektorowej (uruchom raz)
├── requirements.txt
├── data/
│   ├── kodeks.pdf              # ← WRZUĆ TUTAJ
│   └── ruch_drogowy.pdf        # ← WRZUĆ TUTAJ
├── vectorstore/        # generowany automatycznie
└── .streamlit/
    └── secrets.toml    # ← WPISZ TUTAJ TOKEN HF
```

## Uruchomienie lokalne

### 1. Zainstaluj zależności
```bash
pip install -r requirements.txt
```

### 2. Wrzuć PDFy do folderu `data/`
- `kodeks.pdf` — Kodeks postępowania w sprawach o wykroczenia
- `ruch_drogowy.pdf` — Ustawa Prawo o ruchu drogowym

### 3. Aplikacja działa przez Bielika, którego stawiamy lokalnie
SpeakLeash/bielik-4.5b-v3.0-instruct:Q8_0

Pobierz Bielika lokalnie
Pobierz Ollama -> przejdz do terminala i wpisz
    ollama pull SpeakLeash/bielik-4.5b-v3.0-instruct:Q8_0

sprawdź czy działa Bielik i zadaj mu pytanie
ollama run SpeakLeash/bielik-4.5b-v3.0-instruct:Q8_0

Gotowe!

### 4. Zbuduj bazę wektorową (tylko raz!)
```bash
python ingest.py
```

### 5. Uruchom aplikację
```bash
streamlit run app.py
```

## Deploy na Streamlit Cloud
1. Wrzuć kod na GitHub (bez `secrets.toml` i folderu `vectorstore/`)
2. Połącz repo z [share.streamlit.io](https://share.streamlit.io)
3. W ustawieniach aplikacji dodaj secret: `HF_TOKEN`
4. **Uwaga:** na Streamlit Cloud trzeba uruchomić `ingest.py` jako część startu
   lub wrzucić gotowy `vectorstore/` osobno (np. przez Git LFS)
