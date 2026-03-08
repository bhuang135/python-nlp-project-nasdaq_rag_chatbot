@echo off
echo ===============================
echo Starting Nasdaq RAG Chatbot
echo ===============================

call venv\Scripts\activate

echo Launching Streamlit UI...

streamlit run app.py

pause