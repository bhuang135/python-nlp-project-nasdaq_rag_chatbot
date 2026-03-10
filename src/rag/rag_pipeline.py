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
        Streamlit chat UI 對外統一入口。
        先做最小可用版本：直接查 collection，取回文件，再交給 Gemini 摘要。
        """

        conversation_history = conversation_history or []

        # 1) 先查向量資料庫
        results = self.collection.query(
            query_texts=[query],
            n_results=3,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        # 2) 若查不到資料
        if not documents:
            return {
                "answer": "I could not find relevant information in the current knowledge base.",
                "citations": [],
                "company_context": current_company,
                "model_name": self.model_name,
            }

        # 3) 建 context
        context_parts = []
        citations = []

        for idx, doc in enumerate(documents):
            meta = metadatas[idx] if idx < len(metadatas) else {}

            label = f"Source {idx + 1}"
            company = meta.get("company", "Unknown")
            ticker = meta.get("ticker", "N/A")
            source = meta.get("source", "N/A")
            doc_type = meta.get("doc_type", "N/A")
            title = meta.get("title", "Untitled")

            context_parts.append(
                f"[{label}] Company: {company} ({ticker})\nTitle: {title}\nContent: {doc}"
            )

            citations.append(
                {
                    "label": label,
                    "company": company,
                    "ticker": ticker,
                    "source": source,
                    "doc_type": doc_type,
                    "title": title,
                    "snippet": doc[:300],
                }
            )

        context = "\n\n".join(context_parts)

        # 4) 呼叫 Gemini
        prompt = f"""
You are a financial research assistant.

Answer the user's question using ONLY the context below.
If the context is insufficient, say so clearly.
Be concise but helpful.

User question:
{query}

Context:
{context}
"""

        response = self.genai_client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        answer_text = response.text if hasattr(response, "text") else str(response)

        # 5) 嘗試從 citation metadata 帶出 company context
        company_context = current_company
        if metadatas:
            first_meta = metadatas[0]
            if first_meta.get("company") and first_meta.get("ticker"):
                company_context = {
                    "company": first_meta.get("company"),
                    "ticker": first_meta.get("ticker"),
                }

        return {
            "answer": answer_text,
            "citations": citations,
            "company_context": company_context,
            "model_name": self.model_name,
        }
