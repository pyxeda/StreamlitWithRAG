import streamlit as st
from openai import OpenAI
from pinecone_utils import QA_MODEL, streaming_question_answering, get_similar_context


st.title("Retrieval Augment Chat")

# set openai key
client = OpenAI(api_key= st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = QA_MODEL

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        pinecone_context = get_similar_context(prompt)
        response = st.write_stream(streaming_question_answering(prompt, pinecone_context))
    st.session_state.messages.append({"role": "assistant", "content": response})