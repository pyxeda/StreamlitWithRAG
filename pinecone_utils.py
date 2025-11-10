import hashlib
import os
import uuid
from typing import List
from pinecone import Pinecone
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


TEXT_MODEL = "text-embedding-ada-002"
QA_MODEL = "gpt-4o-mini"
NAMESPACE_KEY = "sample-app"


# API keys
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]
os.environ['INDEX_HOST'] = st.secrets["INDEX_HOST"]

# create client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(host=os.environ["INDEX_HOST"])



COMMON_TEMPLATE = """
"Use the following pieces of context to answer the question at the end with human readable answer as a paragraph"
"Please do not use data outside the context to answer any questions. "
"If the answer is not in the given context, just say that you don't have enough context."
"don't try to make up an answer. "
"\n\n"
{context}
"\n\n"
Question: {question}
"n"
"Helpful answer:   "
"""

def get_model():
    model = ChatOpenAI(model=QA_MODEL, api_key=os.environ["OPENAI_API_KEY"])
    return model


def get_openai_embeddings(text: str) -> list[float]:
    # create client
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.embeddings.create(input=f"{text}", model=TEXT_MODEL)

    return response.data[0].embedding


def upsert_embeddings(embeddings, meta_data, namespace_ = NAMESPACE_KEY):
    vector_id = str(uuid.uuid4())
    upsert_response = index.upsert(
    vectors=[
        (vector_id, embeddings, meta_data),
    ],
    namespace=namespace_
    )

    return upsert_response


# function query similar chunks
def query_response(query_embedding, k = 2, namespace_ = NAMESPACE_KEY):
    query_response = index.query(
        namespace=namespace_,
        vector=query_embedding,
        top_k=k,
        include_values=False,
        include_metadata=True,
    )

    return query_response


def content_extractor(similar_data):
    top_values = similar_data["matches"]
    # get the text out
    text_content = [sub_content["metadata"]["text"] for sub_content in top_values]
    return " ".join(text_content)


def upload_to_pinecone(text_document: str, file_name, chunk_size: int = 1000 ) -> None:
    """
    Upload the text content to pinecone

    @params
    text_document: text content needs to upload
    file_name: name of the filed to be included as metadata
    chunk_size: chunk size to split the data

    @return
    None
    """

    MODEL = "text-embedding-ada-002"

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = chunk_size,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    # text splitter
    texts = text_splitter.create_documents([text_document])

    for chunk_document in texts:
        chunk_text_data = chunk_document.page_content
        # get embeddings
        embeddings = get_openai_embeddings(chunk_text_data)
        # create metadata
        meta_creator = {"text": chunk_text_data}
        # store embeddings
        emb_response = upsert_embeddings(embeddings, meta_creator)

    return True


def get_similar_context(question: str):
    # get the query embeddings
    quer_embed_data = get_openai_embeddings(question)

    # query the similar chunks
    similar_chunks = query_response(quer_embed_data)

    # extract the similar text data
    similar_content = content_extractor(similar_chunks)

    return similar_content


def question_answering(query_question, context_text, template=COMMON_TEMPLATE):
    prompt = ChatPromptTemplate.from_template(template)
    model = get_model()
    output_parser = StrOutputParser()

    # create the chain
    chain = prompt | model | output_parser

    # get the answer
    answer = chain.invoke({"context": context_text, "question": query_question})

    return answer


def streaming_question_answering(query_question: str, context_text: str,  template: str = COMMON_TEMPLATE):
    prompt = ChatPromptTemplate.from_template(template)
    model = get_model()
    output_parser = StrOutputParser()

    # create the chain
    chain = prompt | model | output_parser

    # get the answer
    return chain.stream({"context": context_text, "question": query_question})