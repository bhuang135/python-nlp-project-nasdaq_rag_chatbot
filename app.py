import os
import sys
from pathlib import Path

# 將專案中的 src 資料夾加入 Python 搜尋路徑
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag.rag_pipeline import RAGPipeline

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
