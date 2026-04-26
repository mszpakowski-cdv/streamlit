import streamlit as st
from openai import OpenAI
import os

st.set_page_config(layout="wide", page_title="Gemini chatbot app")
st.title("OpenAI chatbot app")
uploaded_file = st.file_uploader(label="Dodaj załącznik")

# api_key, base_url = os.environ["API_KEY"], os.environ["BASE_URL"]
api_key, base_url = st.secrets["API_KEY"], st.secrets["BASE_URL"]
selected_model = "gpt-4o"

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not api_key:
        st.info("Invalid API key.")
        st.stop()
    client = OpenAI(
         # This is the default and can be omitted
        api_key=api_key,
    )

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if uploaded_file is not None:
        client.files.create(
            file=uploaded_file.read().decode("utf-8"),
            purpose="assistants"
        )
        st.session_state.messages.append({"role": "user", "content": uploaded_file})

    uploaded_file = None

    response = client.chat.completions.create(
        model=selected_model,
        messages=st.session_state.messages,
    )

    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
