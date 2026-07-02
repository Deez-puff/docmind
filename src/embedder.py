import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ── Embedding Model ───────────────────────────────────────
# We use sentence-transformers/all-MiniLM-L6-v2
# Why this model?
# - Free, runs locally (no API cost)
# - Small (90MB) but powerful
# - Great for semantic similarity search
# - Industry standard for RAG applications
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    """
    Loads the sentence-transformer embedding model.
    Downloads automatically on first use (~90MB).
    Cached locally after first download.
    """
    print("⏳ Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✅ Embedding model loaded!")
    return embeddings


def create_vectorstore(chunks, pdf_name, persist_dir="data/vectorstore"):
    """
    Converts text chunks into vectors and stores them in ChromaDB.

    How it works:
    1. Each chunk is converted to a 384-dimension vector
    2. Vectors are stored in ChromaDB with metadata
    3. ChromaDB persists to disk so you don't re-embed every time

    chunks: list of text strings
    pdf_name: used as collection name in ChromaDB
    persist_dir: where ChromaDB saves its data
    """
    print(f"\n⏳ Creating vector store for '{pdf_name}'...")

    # ── Convert chunks to LangChain Document objects ──────
    # Documents have 'page_content' (text) and 'metadata'
    documents = [
        Document(
            page_content=chunk,
            metadata={"source": pdf_name, "chunk_id": i}
        )
        for i, chunk in enumerate(chunks)
    ]

    # ── Load embedding model ──────────────────────────────
    embeddings = get_embeddings()

    # ── Create ChromaDB vector store ──────────────────────
    # collection_name = sanitized pdf name (no spaces/dots)
    collection_name = pdf_name.replace(" ", "_").replace(".", "_").lower()

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir
    )

    print(f"✅ Vector store created with {len(documents)} chunks!")
    print(f"✅ Saved to '{persist_dir}'")

    return vectorstore


def load_vectorstore(pdf_name, persist_dir="data/vectorstore"):
    """
    Loads an existing ChromaDB vector store from disk.
    Much faster than re-embedding — reuses saved vectors.
    """
    print(f"⏳ Loading existing vector store for '{pdf_name}'...")

    embeddings = get_embeddings()
    collection_name = pdf_name.replace(" ", "_").replace(".", "_").lower()

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir
    )

    print(f"✅ Vector store loaded!")
    return vectorstore


def vectorstore_exists(pdf_name, persist_dir="data/vectorstore"):
    """
    Checks if a vector store already exists for a given PDF.
    Avoids re-embedding the same PDF multiple times.
    """
    collection_name = pdf_name.replace(" ", "_").replace(".", "_").lower()
    chroma_path = os.path.join(persist_dir, collection_name)
    return os.path.exists(chroma_path)