import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

_vector_store = None
_retriever = None


# ---------------------------------------
# Embedding selector
# ---------------------------------------


def get_embeddings():

    openai_key = None

    if openai_key:
        print("Using OpenAI Embeddings")
        return OpenAIEmbeddings()

    print("Using HuggingFace Embeddings")

    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ---------------------------------------
# Build / Update vector store
# ---------------------------------------


def build_vector_store(file_path: str):

    global _vector_store
    global _retriever

    print(f"Processing file: {file_path}")

    docs = TextLoader(file_path, encoding="utf-8").load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings()

    if _vector_store is None:

        print("Creating FAISS vector store")

        _vector_store = FAISS.from_documents(chunks, embeddings)

    else:

        print("Updating existing vector store")

        _vector_store.add_documents(chunks)

    _retriever = _vector_store.as_retriever(search_kwargs={"k": 3})

    print("Vector store ready")

    return _retriever


# ---------------------------------------
# Access retriever
# ---------------------------------------


def get_retriever():
    return _retriever
