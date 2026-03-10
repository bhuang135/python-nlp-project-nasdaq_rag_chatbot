import os
import sys
import streamlit as st
import chromadb

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from src.rag.rag_pipeline import RAGPipeline
from build_index import build_index


CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "nasdaq_docs"


def collection_exists(chroma_dir: str, collection_name: str) -> bool:
    try:
        client = chromadb.PersistentClient(path=chroma_dir)
        collections = client.list_collections()

        for c in collections:
            if hasattr(c, "name"):
                if c.name == collection_name:
                    return True
            elif c == collection_name:
                return True

        return False

    except Exception:
        return False


def ensure_index():
    os.makedirs(CHROMA_DIR, exist_ok=True)

    if not collection_exists(CHROMA_DIR, COLLECTION_NAME):
        with st.spinner("Building vector database for first launch..."):
            build_index(
                chroma_dir=CHROMA_DIR,
                collection_name=COLLECTION_NAME,
                limit=30
            )


st.set_page_config(page_title="Nasdaq RAG Chatbot", layout="wide")

st.title("Nasdaq RAG Chatbot")
st.caption(
    "Ask about any Nasdaq company using ticker, company name, natural language, or follow-up questions."
)

# session state
if "history" not in st.session_state:
    st.session_state.history = []

if "current_company" not in st.session_state:
    st.session_state.current_company = None

# initialize rag
if "rag" not in st.session_state:
    ensure_index()
    st.session_state.rag = RAGPipeline(
        chroma_dir=CHROMA_DIR,
        collection_name=COLLECTION_NAME
    )


query = st.chat_input("Example: What is AAPL")

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
                with st.expander("Sources"):
                    for c in citations:
                        st.markdown(
                            f"**{c['label']}** — {c['company']} ({c['ticker']})"
                        )
                        st.write(c["snippet"])
