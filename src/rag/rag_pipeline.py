import os
import chromadb
from google import genai
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

from src.resolver.company_resolver import CompanyResolver

load_dotenv()


def get_gemini_api_key():
    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        return api_key

    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    raise ValueError("GEMINI_API_KEY is not set.")


class RAGPipeline:

    def __init__(
        self,
        chroma_dir: str = "chroma_db",
        collection_name: str = "nasdaq_docs",
        preferred_model: str = "gemini-2.5-flash",
    ):

        api_key = get_gemini_api_key()

        self.genai_client = genai.Client(api_key=api_key)
        self.model_name = preferred_model

        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.client = chromadb.PersistentClient(path=chroma_dir)

        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
            )
        except Exception as e:
            raise RuntimeError(
                f"Collection '{collection_name}' not found. "
                f"Index may not be built yet. Original error: {e}"
            )

        self.resolver = CompanyResolver()
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
