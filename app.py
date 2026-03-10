import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

import streamlit as st
import chromadb

from src.rag.rag_pipeline import RAGPipeline
from build_index import build_index


CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "nasdaq_docs"


def collection_exists(chroma_dir: str, collection_name: str) -> bool:
    """
    檢查指定的 Chroma collection 是否存在。
    """
    try:
        client = chromadb.PersistentClient(path=chroma_dir)
        collections = client.list_collections()

        # 不同版本 chromadb 可能回傳 object 或 string
        for c in collections:
            if hasattr(c, "name"):
                if c.name == collection_name:
                    return True
            elif c == collection_name:
                return True

        return False

    except Exception as e:
        st.warning(f"Failed to check Chroma collection: {e}")
        return False


def ensure_index():
    """
    若 collection 不存在，則建立 index。
    """
    if not os.path.exists(CHROMA_DIR):
        os.makedirs(CHROMA_DIR, exist_ok=True)

    if not collection_exists(CHROMA_DIR, COLLECTION_NAME):
        with st.spinner("Vector database not found. Building index for first launch..."):
            build_index(
                chroma_dir=CHROMA_DIR,
                collection_name=COLLECTION_NAME
            )
        st.success("Index built successfully.")


st.set_page_config(page_title="Nasdaq RAG Chatbot", layout="wide")
st.title("Nasdaq RAG Chatbot")
st.write("Ask about any Nasdaq company using ticker, company name, natural language, or follow-up questions.")

if "rag" not in st.session_state:
    ensure_index()
    st.session_state.rag = RAGPipeline(
        chroma_dir=CHROMA_DIR,
        collection_name=COLLECTION_NAME
    )

query = st.text_input("Enter your question:")

if query:
    with st.spinner("Thinking..."):
        answer = st.session_state.rag.answer_question(query)
    st.write(answer)



st.set_page_config(page_title="Nasdaq RAG Chatbot", layout="wide")
st.title("Nasdaq RAG Chatbot")
st.caption("Ask about any Nasdaq company using ticker, company name, natural language, or follow-up questions.")

if "history" not in st.session_state:
    st.session_state.history = []

if "current_company" not in st.session_state:
    st.session_state.current_company = None

if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline()

query = st.chat_input("Example: Can you summarize AAPL business and financial condition?")

if query:
    st.session_state.history.append(("user", query))

    try:
        result = st.session_state.rag.answer(
            query=query,
            conversation_history=st.session_state.history[:-1],
            current_company=st.session_state.current_company,
        )

        if result.get("company_context"):
            st.session_state.current_company = result["company_context"]

        st.session_state.history.append(
            (
                "assistant",
                {
                    "answer": result["answer"],
                    "citations": result["citations"],
                    "company_context": result["company_context"],
                    "model_name": result.get("model_name"),
                },
            )
        )
    except Exception as e:
        st.session_state.history.append(
            (
                "assistant",
                {
                    "answer": f"System error: {e}",
                    "citations": [],
                    "company_context": None,
                    "model_name": None,
                },
            )
        )

for role, payload in st.session_state.history:
    with st.chat_message(role):
        if role == "user":
            st.write(payload)
        else:
            st.write(payload["answer"])

            if payload.get("model_name"):
                st.caption(f"Model used: {payload['model_name']}")

            if payload.get("company_context"):
                cc = payload["company_context"]
                st.caption(f"Resolved company: {cc['company']} ({cc['ticker']})")

            citations = payload.get("citations", [])
            if citations:
                with st.expander("Sources / Citations"):
                    for c in citations:
                        st.markdown(
                            f"**{c['label']}** — {c['company']} ({c['ticker']}) | "
                            f"{c['source']} | {c['doc_type']} | {c['title']}"
                        )
                        st.write(c["snippet"])
