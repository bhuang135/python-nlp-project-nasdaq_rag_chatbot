
# Nasdaq RAG Chatbot


---

# Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot designed to answer questions about **Nasdaq listed companies** using real financial data sources such as:

- SEC filings (10-K, 10-Q)
- SEC XBRL financial facts
- Yahoo Finance company profiles
- Financial snapshot data
- Recent financial news

The system combines:

- Vector search (ChromaDB)
- Hybrid retrieval (semantic + BM25)
- Cross-encoder reranking
- Multi-query retrieval
- LLM answer generation (Gemini)
- Conversation memory
- Source citations

The chatbot behaves similarly to ChatGPT but **only answers using retrieved documents** and **never fabricates financial information**.

---

# System Architecture

User Question
      |
      v
Company Resolver
      |
      v
Query Expansion + Multi Query Generation
      |
      v
Vector Search (ChromaDB)
      |
      v
Hybrid Ranking (Semantic + BM25)
      |
      v
Cross Encoder Re-ranking
      |
      v
Context Assembly
      |
      v
LLM (Gemini)
      |
      v
Grounded Answer + Citations

---

# Features

## 1. Company Alias Resolution

The chatbot supports multiple ways to refer to the same company.

Examples:

Apple  
Apple Inc  
AAPL  
$AAPL  

All resolve to:

Ticker: AAPL  
Company: Apple Inc.

---

## 2. Natural Language Queries

The system understands natural questions such as:

Summarize Apple's business model  
Explain Nvidia's core products and services  
What is Apple's latest revenue and net income  
Show Tesla's financial condition based on recent data  

---

## 3. Multi Company Comparison

The chatbot supports comparison questions:

Compare Apple and Microsoft  
Which company is more profitable Nvidia or AMD  

The system automatically detects both companies and retrieves information separately before generating a comparison answer.

---

## 4. Hybrid Search

Retrieval combines:

Semantic Search  
Vector embeddings (SentenceTransformer)

Keyword Search  
BM25 ranking

This improves recall and reduces retrieval failures.

---

## 5. Re-ranking

Initial top 20 chunks are reranked using:

Cross Encoder Model  
cross-encoder/ms-marco-MiniLM-L-6-v2

Top 5 results are sent to the LLM.

---

## 6. Multi Query Retrieval

User questions are expanded into multiple search queries using:

Rule based expansion  
LLM query rewriting

Example:

User question:

Summarize Apple's business model

Generated retrieval queries:

company profile  
business overview  
products and services  
Apple business overview  
Apple company profile  

---

## 7. Grounded Answer Generation

The LLM must follow strict rules:

- Answer only using retrieved context
- No hallucinated facts
- Include source citations

Example output:

Apple generates most revenue from iPhone, followed by services and Mac products [S1][S2].

Sources Used: [S1], [S2]

---

# Data Sources

The system ingests data from:

SEC EDGAR API  
XBRL Company Facts  
Yahoo Finance  
Financial News APIs

Document types stored in the vector database:

yf_company_profile  
yf_financial_snapshot  
yf_news  
sec_entity_profile  
sec_companyfacts  
sec_recent_filings

---

# Project Structure

project/

src/

resolver/
company_resolver.py

rag/
rag_pipeline.py

data/
companies.csv

chroma_db/

scripts/
build_index.py

app.py

README.md

---

# Installation

## 1 Install Dependencies

pip install -r requirements.txt

Key libraries:

chromadb  
sentence-transformers  
rank-bm25  
google-generativeai  
rapidfuzz  
pandas  

---

# Environment Variables

Create `.env` file

GEMINI_API_KEY=your_gemini_key  
SEC_API_EMAIL=your_email

SEC requires a valid email in headers for API access.

---

# Build the Vector Database

Run:

python scripts/build_index.py

This step:

1 Downloads company metadata  
2 Fetches SEC filings  
3 Fetches financial snapshots  
4 Processes documents  
5 Splits documents into chunks  
6 Generates embeddings  
7 Stores them in ChromaDB  

Expected output:

chroma_db/

---

# Running the Chatbot

Run:

streamlit run app.py

Open browser:

http://localhost:8501

---

# Example Queries

Business Questions

Summarize Apple's business model  
Explain Nvidia's core products and services  

Financial Questions

What is Apple's latest revenue and net income  
Show Tesla's financial condition  

News Questions

What are the latest news about Tesla  
Summarize recent news about Microsoft  

Comparison Questions

Compare Apple and Microsoft  
Which company is more profitable Nvidia or AMD  

---

# Evaluation Metrics

The system can be evaluated using:

Retrieval Recall  
Relevant document retrieved in top K

Answer Groundedness  
All claims supported by sources

Faithfulness  
No hallucinated facts

Citation Accuracy  
Correct document referenced

---

# Deployment Options

The system can be deployed using:

Streamlit Cloud  
Docker  
Render  
AWS EC2

Recommended for coursework:

Streamlit Cloud

---

# Docker Deployment

Build image:

docker build -t nasdaq-rag-chatbot .

Run container:

docker run -p 8501:8501 nasdaq-rag-chatbot

---

# One Click Run

After installation:

python scripts/build_index.py  
streamlit run app.py

---

# Limitations

Answers depend on retrieved documents.

If relevant documents are missing or outdated, the system may respond:

"I don't have enough information in the retrieved documents to answer that."

---

# Future Improvements

Realtime financial data ingestion  
Better financial comparison models  
Structured financial table extraction  
Advanced evaluation benchmarks  

---

# License

Educational Use Only

---

# Contact

Bright Huang  
MSBA Candidate  
UC Irvine  

shengwh4@uci.edu
