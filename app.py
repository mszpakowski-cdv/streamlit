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

# Adres lokalnego serwera Ollama oraz nazwa modelu Bielik
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "SpeakLeash/bielik-4.5b-v3.0-instruct:Q8_0"

SYSTEM_PROMPT = """Jesteś asystentem prawnym dla policjantów drogówki w Polsce.
Twoim zadaniem jest analiza opisanych sytuacji drogowych i wskazanie:
1. Które przepisy zostały naruszone (z dokładnym numerem artykułu)
2. Jaka kara lub mandat grozi sprawcy
3. Tryb postępowania (mandat, wniosek o ukaranie itd.)

Odpowiadaj zwięźle, konkretnie i po polsku.
Cytuj numery artykułów. Jeśli kontekst nie zawiera odpowiedzi — powiedz to wprost.
Nie wymyślaj przepisów."""

# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Agent Kodeksu Drogowego",
    page_icon="🚔",
    layout="centered"
)

st.title("🚔 Agent Kodeksu Drogowego")
st.caption("System wsparcia prawnego dla policjantów drogówki")

# Sprawdź czy vectorstore istnieje
if not os.path.exists("vectorstore/index.faiss"):
    st.error(
        "⚠️ Baza wiedzy nie istnieje. Uruchom najpierw: `python ingest.py`"
    )
    st.stop()


# ── Funkcja wywołująca Bielika przez Ollama ──────────────────────────────────

def zapytaj_bielika(system_prompt: str, user_prompt: str) -> str:
    """Wysyła zapytanie do lokalnego serwera Ollama i zwraca odpowiedź modelu."""
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

# Wyświetl historię
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Opisz sytuację na drodze..."):

    # Pokaż wiadomość użytkownika
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        # Guardrails
        if not is_traffic_related(prompt):
            response = (
                "⚠️ To zapytanie nie dotyczy przepisów ruchu drogowego ani wykroczeń. "
                "Opisz konkretną sytuację drogową, a pomogę wskazać właściwe przepisy."
            )
            st.markdown(response)

        else:
            # RAG — pobierz kontekst
            with st.spinner("🔍 Przeszukuję kodeks..."):
                chunks = retrieve(prompt, k=5)
                context = format_context(chunks)

            # Buduj prompt z kontekstem
            full_user_prompt = f"""Przepisy prawne (kontekst):
{context}

Sytuacja zgłoszona przez policjanta:
{prompt}

Wskaż naruszone przepisy i grożące kary."""

            # Wywołanie Bielika przez Ollama
            with st.spinner("⚖️ Analizuję przepisy..."):
                try:
                    response = zapytaj_bielika(SYSTEM_PROMPT, full_user_prompt)
                except requests.exceptions.ConnectionError:
                    response = (
                        "❌ Nie mogę połączyć się z Ollamą. "
                        "Upewnij się, że Ollama jest uruchomiona (ikonka lamy na pasku) "
                        "i model jest pobrany."
                    )
                except Exception as e:
                    response = f"❌ Błąd połączenia z modelem: {e}"

            st.markdown(response)

            # Pokaż źródła
            with st.expander("📄 Źródła (znalezione fragmenty przepisów)"):
                for i, chunk in enumerate(chunks, 1):
                    source_label = {
                        "kodeks_wykroczen": "Kodeks postępowania w sprawach o wykroczenia",
                        "prawo_o_ruchu_drogowym": "Prawo o ruchu drogowym",
                    }.get(chunk["source"], chunk["source"])
                    st.markdown(f"**{i}. {source_label} — {chunk['article']}**")
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
