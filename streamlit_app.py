import streamlit as st
from openai import OpenAI
import io
import faiss
import numpy as np
from pypdf import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(layout="wide", page_title="OpenAI chatbot app")
st.title("OpenAI chatbot app")
uploaded_file = st.file_uploader(label="Dodaj załącznik")

api_key, base_url = st.secrets["API_KEY"], st.secrets["BASE_URL"]
selected_model = "gpt-4o"

class FAISSIndex:
    def __init__(self, faiss_index, metadata):
        self.index = faiss_index
        self.metadata = metadata

    def similarity_search(self, query, k=3):
        D, I = self.index.search(query, k)
        results = []
        for idx in I[0]:
            results.append(self.metadata[idx])
        return results

embed_model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model_kwargs = {"device": "cpu", "trust_remote_code": True}

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=embed_model_id, model_kwargs=model_kwargs)


def chunk_text(text, size=800, overlap=100):
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += size - overlap
    return chunks


def create_index(documents):
    embeddings = get_embeddings()
    texts = [doc["text"] for doc in documents]
    metadata = documents

    embeddings_matrix = [embeddings.embed_query(text) for text in texts]
    embeddings_matrix = np.array(embeddings_matrix).astype("float32")

    index = faiss.IndexFlatL2(embeddings_matrix.shape[1])
    index.add(embeddings_matrix)

    return FAISSIndex(index, metadata)


def retrieve_docs(query, faiss_index, k=3):
    embeddings = get_embeddings()
    query_embedding = np.array([embeddings.embed_query(query)]).astype("float32")
    results = faiss_index.similarity_search(query_embedding, k=k)
    return results


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?."}]
if "faiss_index" not in st.session_state:
    st.session_state["faiss_index"] = None
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
        chunks = chunk_text(text)
        documents = [{"filename": uploaded_file.name, "text": c} for c in chunks]
        st.session_state.faiss_index = create_index(documents)
        st.session_state.indexed_filename = uploaded_file.name

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    messages_for_api = list(st.session_state.messages)
    if st.session_state.faiss_index is not None:
        relevant = retrieve_docs(prompt, st.session_state.faiss_index)
        context = "\n\n---\n\n".join(doc["text"] for doc in relevant)
        messages_for_api.append({
            "role": "system",
            "content": f"Answer only based on the user's file if attached. Use the following context from the user's document to answer:\n\n{context}",
        })

    response = client.chat.completions.create(
        model=selected_model,
        messages=messages_for_api,
    )

    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
