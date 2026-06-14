import streamlit as st
import io
from threading import Thread
import faiss
import numpy as np
import torch
from pypdf import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

st.set_page_config(layout="wide", page_title="Bielik chatbot app")
st.title("Bielik chatbot app")
uploaded_file = st.file_uploader(label="Dodaj załącznik")

MODEL_ID = "speakleash/Bielik-PL-11B-v3.0-Instruct"

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


@st.cache_resource
def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    return tokenizer, model


def generate_response(messages, max_new_tokens=1024):
    tokenizer, model = load_model()
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    generation_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    for token in streamer:
        yield token


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
    st.session_state["messages"] = [{"role": "assistant", "content": "W czym mogę pomóc?"}]
if "faiss_index" not in st.session_state:
    st.session_state["faiss_index"] = None
    st.session_state["indexed_filename"] = None

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if uploaded_file is not None and uploaded_file.name != st.session_state.indexed_filename:
        reader = PdfReader(io.BytesIO(uploaded_file.read()))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        chunks = chunk_text(text)
        documents = [{"filename": uploaded_file.name, "text": c} for c in chunks]
        st.session_state.faiss_index = create_index(documents)
        st.session_state.indexed_filename = uploaded_file.name

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Build the conversation for the model, dropping the leading assistant
    # greeting so the dialog starts with a user turn (chat templates require it).
    messages_for_api = list(st.session_state.messages)
    while messages_for_api and messages_for_api[0]["role"] == "assistant":
        messages_for_api.pop(0)

    if st.session_state.faiss_index is not None:
        relevant = retrieve_docs(prompt, st.session_state.faiss_index)
        context = "\n\n---\n\n".join(doc["text"] for doc in relevant)
        messages_for_api.insert(0, {
            "role": "system",
            "content": f"Odpowiadaj wyłącznie na podstawie załączonego pliku użytkownika. Wykorzystaj poniższy kontekst z dokumentu użytkownika, aby udzielić odpowiedzi:\n\n{context}",
        })

    with st.chat_message("assistant"):
        msg = st.write_stream(generate_response(messages_for_api))
    st.session_state.messages.append({"role": "assistant", "content": msg})
