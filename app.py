"""
app.py — główna aplikacja Streamlit
Agent Kodeksu Drogowego dla policjantów drogówki

Model: Bielik uruchomiony lokalnie przez Ollama (http://localhost:11434)
"""

import os
import requests
import streamlit as st
from rag import retrieve, format_context
from guardrails import is_traffic_related

# ── Konfiguracja ──────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "hf.co/gaianet/Bielik-4.5B-v3.0-Instruct-GGUF:Q6_K"

# Ile fragmentów przepisów trafia do modelu po rerankingu (mniej, ale trafniej)
TOP_K = 5

SYSTEM_PROMPT = """Jesteś asystentem prawnym dla polskich policjantów drogówki. Oceniasz, czy opisana
sytuacja drogowa stanowi naruszenie przepisów, na podstawie DOSTARCZONYCH przepisów.

STANDARD OCENY:
Stosujesz standard administracyjny (jak w sprawach o wykroczenia drogowe), NIE karny.
NIE wymagaj pewności „ponad wszelką wątpliwość". Naruszenie stwierdzasz, gdy z faktów
i treści przepisu wynika ono w sposób przeważający (bardziej prawdopodobne niż nie).
Przeoczenie realnego naruszenia jest BŁĘDEM TAK SAMO POWAŻNYM jak uznanie czystej
sytuacji za naruszenie. Nie faworyzuj żadnej z odpowiedzi z góry — oceniaj fakty.

SPOSÓB PRACY:
1. Opieraj się WYŁĄCZNIE na przepisach z sekcji „PRZEPISY (KONTEKST)". Cytuj dokładny
   numer artykułu i ustępu z tego kontekstu.
2. Dla każdego istotnego przepisu wykonaj ślad faktów:
   a) Przytocz, co przepis NAKAZUJE lub ZABRANIA (kto, komu, co, w którą stronę, pod
      jakim warunkiem, z jakim wyjątkiem). Czytaj dosłownie i nie odwracaj kierunku
      reguły (np. „polecenia osoby kierującej ruchem MAJĄ pierwszeństwo przed sygnałami
      świetlnymi" — nie na odwrót).
   b) Wypisz fakty z opisu — co kierujący FAKTYCZNIE zrobił lub czego zaniechał. Nie
      dopisuj zachowań, których w opisie nie ma (np. nie zakładaj, że „kierujący
      ustąpił", jeśli opis tego nie mówi).
   c) Porównaj: czy fakty spełniają obowiązek, czy go naruszają. Zaniechanie obowiązku
      (np. niezachowanie szczególnej ostrożności) jest naruszeniem tak samo jak
      działanie zakazane.
3. Traktuj obowiązek jako całość — nie rozbijaj jednego obowiązku na osobne reguły, by
   każdą z osobna oddalić.
4. Zanim orzekniesz naruszenie, sprawdź WYJĄTKI — czy treść przepisu nie przewiduje
   okoliczności wyłączającej naruszenie (np. pojazd uprzywilejowany z włączonymi
   sygnałami; zachowanie wprost dopuszczone przez przepis jako wariant; pierwszeństwo po
   stronie kierującego). Jeśli przepis przewiduje taki wyjątek i tu on zachodzi — to
   „Brak naruszenia".
5. Werdykt: jeśli fakty naruszają przepis i nie zachodzi wyjątek — napisz, że doszło do
   naruszenia, i wskaż artykuł; jeśli naruszenie nie wynika — napisz wprost „Brak
   naruszenia" i wyjaśnij, którego warunku nie spełniono; jeśli przepisy nie wystarczają
   — powiedz to otwarcie.

KARA, MANDAT, KWALIFIKACJA:
Jeśli w sekcji „PRZEPISY (KONTEKST)" jest pozycja taryfikatora mandatów odpowiadająca
czynowi — podaj kwalifikację Kodeksu wykroczeń (np. „art. 92a § 1 k.w.") oraz kwotę
grzywny dokładnie tak, jak w taryfikatorze, i zaznacz, że pochodzi z taryfikatora.
Jeśli odpowiedniej pozycji NIE ma w kontekście — NIE wymyślaj artykułu k.w., kwoty ani
punktów. Napisz, że nie wynika to z dostarczonych przepisów, albo umieść pod flagą
„⚠️ Uwaga: poniższe nie wynika z dostarczonych przepisów, podaję z wiedzy ogólnej:".

STRUKTURA ODPOWIEDZI:
1. Analiza — ślad faktów (obowiązek → fakty → wniosek)
2. Ocena — naruszone przepisy ruchu (z numerami) ALBO wyraźne „Brak naruszenia"
3. Kara / mandat — kwalifikacja k.w. i kwota grzywny z taryfikatora w kontekście;
   jeśli brak pozycji, zaznacz to (nie zgaduj kwoty)
4. Tryb postępowania — tylko jeśli wynika z przepisów KPW w kontekście

Odpowiadaj zwięźle, konkretnie i po polsku."""

# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Agent Kodeksu Drogowego",
    page_icon="🚔",
    layout="centered"
)

st.title("🚔 Agent Kodeksu Drogowego")
st.caption("System wsparcia prawnego dla policjantów drogówki")

if not os.path.exists("vectorstore/index.faiss"):
    st.error("⚠️ Baza wiedzy nie istnieje. Uruchom najpierw: `python ingest.py`")
    st.stop()


# ── Wywołanie Bielika przez Ollama ───────────────────────────────────────────

def zapytaj_bielika(system_prompt: str, user_prompt: str) -> str:
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


# Historia rozmowy
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Opisz sytuację na drodze..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        if not is_traffic_related(prompt):
            response = (
                "⚠️ To zapytanie nie dotyczy przepisów ruchu drogowego ani wykroczeń. "
                "Opisz konkretną sytuację drogową, a pomogę wskazać właściwe przepisy."
            )
            st.markdown(response)

        else:
            with st.spinner("🔍 Przeszukuję kodeks..."):
                chunks = retrieve(prompt, k=TOP_K)
                context = format_context(chunks)

            full_user_prompt = f"""PRZEPISY (KONTEKST):
{context}

SYTUACJA ZGŁOSZONA PRZEZ POLICJANTA:
{prompt}

Oceń tę sytuację: wykonaj ślad faktów dla każdego istotnego przepisu (co przepis
nakazuje/zabrania → co kierujący faktycznie zrobił → wniosek), podaj werdykt
(„naruszenie" + artykuł, albo „Brak naruszenia"), a kwalifikację k.w. i kwotę mandatu
podaj wyłącznie z pozycji taryfikatora obecnej w kontekście — nie zgaduj."""

            with st.spinner("⚖️ Analizuję przepisy..."):
                try:
                    response = zapytaj_bielika(SYSTEM_PROMPT, full_user_prompt)
                except requests.exceptions.ConnectionError:
                    response = (
                        "❌ Nie mogę połączyć się z Ollamą. "
                        "Upewnij się, że Ollama jest uruchomiona (ikonka lamy na pasku)."
                    )
                except Exception as e:
                    response = f"❌ Błąd połączenia z modelem: {e}"

            st.markdown(response)

            with st.expander("📄 Źródła (znalezione fragmenty przepisów)"):
                for i, chunk in enumerate(chunks, 1):
                    source_label = {
                        "kpw": "Kodeks postępowania w sprawach o wykroczenia",
                        "prawo_o_ruchu_drogowym": "Prawo o ruchu drogowym",
                        "taryfikator": "Taryfikator mandatów",
                    }.get(chunk["source"], chunk["source"])
                    score = chunk.get("score")
                    score_label = f" · trafność {score:.2f}" if score is not None else ""
                    st.markdown(f"**{i}. {source_label} — {chunk['article']}**{score_label}")
                    st.text(chunk["text"][:300] + "...")
                    st.divider()

        st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar
with st.sidebar:
    st.header("ℹ️ Instrukcja")
    st.markdown("""
Opisz sytuację drogową naturalnym językiem, np.:

> *„Kierowca jechał 80 km/h w terenie zabudowanym, nie miał zapiętych pasów i rozmawiał przez telefon"*

Agent wskaże:
- 📌 naruszone artykuły
- 💰 wysokość mandatu
- 📋 tryb postępowania
""")
    st.divider()
    st.caption(f"Model: {MODEL}")
    st.caption("Uruchamiany lokalnie przez Ollama")
    st.divider()
    if st.button("🗑️ Wyczyść historię"):
        st.session_state.messages = []
        st.rerun()