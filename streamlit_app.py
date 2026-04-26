import streamlit as st
from openai import OpenAI
import io
import numpy as np
from pypdf import PdfReader

st.set_page_config(layout="wide", page_title="OpenAI chatbot app")
st.title("OpenAI chatbot app")
uploaded_file = st.file_uploader(label="Dodaj załącznik")

api_key, base_url = st.secrets["API_KEY"], st.secrets["BASE_URL"]
selected_model = "gpt-4o"


def chunk_text(text, size=800, overlap=100):
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += size - overlap
    return chunks

def embed_texts(client, texts):
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([d.embedding for d in resp.data])

def retrieve(client, query, chunks, embeddings, k=3):
    q = embed_texts(client, [query])[0]
    sims = embeddings @ q / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q))
    return [chunks[i] for i in np.argsort(sims)[-k:][::-1]]


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?."}]
if "rag_chunks" not in st.session_state:
    st.session_state["rag_chunks"] = []
    st.session_state["rag_embeddings"] = None
    st.session_state["indexed_filename"] = None

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not api_key:
        st.info("Invalid API key.")
        st.stop()
    client = OpenAI(api_key=api_key)

    if uploaded_file is not None and uploaded_file.name != st.session_state.indexed_filename:
        reader = PdfReader(io.BytesIO(uploaded_file.read()))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        st.session_state.rag_chunks = chunk_text(text)
        st.session_state.rag_embeddings = embed_texts(client, st.session_state.rag_chunks)
        st.session_state.indexed_filename = uploaded_file.name

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    messages_for_api = list(st.session_state.messages)
    if st.session_state.rag_chunks:
        relevant = retrieve(
            client, prompt,
            st.session_state.rag_chunks,
            st.session_state.rag_embeddings,
        )
        context = "\n\n---\n\n".join(relevant)
        messages_for_api.insert(-1, {
            "role": "system",
            "content": f"Use the following context from the user's document to answer:\n\n{context}",
        })

    response = client.chat.completions.create(
        model=selected_model,
        messages=messages_for_api,
    )

    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
