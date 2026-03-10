import os
import chromadb
from chromadb.utils import embedding_functions


def build_index(
    chroma_dir: str = "chroma_db",
    collection_name: str = "nasdaq_docs",
    limit: int | None = 100,
):

    os.makedirs(chroma_dir, exist_ok=True)

    client = chromadb.PersistentClient(path=chroma_dir)

    existing = [c.name if hasattr(c, "name") else c for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(name=collection_name)

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_function,
    )

    docs = [
        {
            "ticker": "AAPL",
            "company": "Apple",
            "text": "Apple Inc designs consumer electronics including iPhone, Mac, and services ecosystem.",
        },
        {
            "ticker": "MSFT",
            "company": "Microsoft",
            "text": "Microsoft provides cloud services Azure, enterprise software, and productivity tools.",
        },
        {
            "ticker": "NVDA",
            "company": "NVIDIA",
            "text": "NVIDIA develops GPUs, AI computing platforms, and accelerated data center hardware.",
        },
    ]

    if limit:
        docs = docs[:limit]

    for i, d in enumerate(docs):

        collection.add(
            documents=[d["text"]],
            ids=[f"doc_{i}"],
            metadatas=[
                {
                    "ticker": d["ticker"],
                    "company": d["company"],
                    "source": "demo",
                    "doc_type": "summary",
                    "title": f"{d['company']} summary",
                }
            ],
        )

    print("Vector index built successfully.")
