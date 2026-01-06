# run_ingestion.py
import os
import sys
from unittest.mock import MagicMock

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Mock torch
sys.modules["torch"] = MagicMock()
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.utils"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["transformers"].PreTrainedTokenizerBase = MagicMock()

load_dotenv()


def run_ingestion():
    print("Starting ingestion...")

    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    print("Loading documents...")
    docs = [WebBaseLoader(url).load() for url in urls]
    doc_list = [item for sublist in docs for item in sublist]
    print(f"Loaded {len(doc_list)} documents")

    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(doc_list)
    print(f"Created {len(splits)} chunks")

    print("Creating embeddings...")
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
    )

    print("Creating FAISS vector store...")
    vector_store = FAISS.from_documents(splits, embeddings)

    print("Saving FAISS index...")
    vector_store.save_local("./faiss_index")

    print("✓ Ingestion complete! FAISS index saved to ./faiss_index")


if __name__ == "__main__":
    run_ingestion()
