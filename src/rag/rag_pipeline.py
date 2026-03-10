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

    def answer(self, query, conversation_history=None, current_company=None):
        """
        Streamlit chat UI 用的統一介面。
        若原本專案主方法叫 answer_question()，這裡做一層包裝。
        """
    
        conversation_history = conversation_history or []
    
        # 情況 1：如果你原本有 answer_question()
        if hasattr(self, "answer_question"):
            raw = self.answer_question(query)
    
            # 若原本只回傳字串，包成前端需要的格式
            if isinstance(raw, str):
                return {
                    "answer": raw,
                    "citations": [],
                    "company_context": current_company,
                    "model_name": getattr(self, "model_name", None),
                }
    
            # 若原本已經是 dict，就補齊欄位
            if isinstance(raw, dict):
                return {
                    "answer": raw.get("answer", ""),
                    "citations": raw.get("citations", []),
                    "company_context": raw.get("company_context", current_company),
                    "model_name": raw.get("model_name", getattr(self, "model_name", None)),
                }
    
        # 情況 2：如果你原本有 ask()
        if hasattr(self, "ask"):
            raw = self.ask(query)
    
            if isinstance(raw, str):
                return {
                    "answer": raw,
                    "citations": [],
                    "company_context": current_company,
                    "model_name": getattr(self, "model_name", None),
                }
    
            if isinstance(raw, dict):
                return {
                    "answer": raw.get("answer", ""),
                    "citations": raw.get("citations", []),
                    "company_context": raw.get("company_context", current_company),
                    "model_name": raw.get("model_name", getattr(self, "model_name", None)),
                }
    
        raise AttributeError(
            "RAGPipeline has neither 'answer()', 'answer_question()', nor 'ask()'. "
            "Please expose one main response method."
        )
