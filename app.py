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
MODEL = "SpeakLeash/bielik-4.5b-v3.0-instruct:Q8_0"

# Ile fragmentów przepisów pobieramy z bazy (więcej = większa szansa na trafny artykuł)
TOP_K = 8

SYSTEM_PROMPT = """Jesteś asystentem prawnym dla polskich policjantów drogówki.
Analizujesz opisane sytuacje drogowe i wskazujesz naruszone przepisy oraz grożące kary.

ZASADY ODPOWIEDZI:
1. W pierwszej kolejności opieraj się WYŁĄCZNIE na przepisach podanych w sekcji
   "PRZEPISY (KONTEKST)". Cytuj dokładne numery artykułów z tego kontekstu.
2. Jeśli w dostarczonych przepisach NIE MA podstawy do oceny sytuacji, a znasz
   odpowiedź z własnej wiedzy — możesz jej udzielić, ale MUSISZ wyraźnie oznaczyć
   to ostrzeżeniem w osobnej sekcji:
   "⚠️ Uwaga: poniższe nie wynika z dostarczonych przepisów, podaję z wiedzy ogólnej:"
3. NIGDY nie mieszaj obu źródeł. Najpierw to, co wynika z kontekstu (z numerami
   artykułów), a dopiero potem — jeśli trzeba — osobno to, co dopowiadasz z wiedzy własnej.
4. Nie podawaj numeru artykułu jako pochodzącego z kontekstu, jeśli go tam nie ma.

STRUKTURA ODPOWIEDZI:
1. Naruszone przepisy (z numerami artykułów z kontekstu)
2. Grożąca kara / mandat
3. Tryb postępowania

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

Wskaż naruszone przepisy i grożące kary zgodnie z zasadami odpowiedzi."""

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