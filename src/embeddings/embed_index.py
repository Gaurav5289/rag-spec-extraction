import os
from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.utils.config import INDEX_DIR
from src.utils.logger import logger


# Embedding model (lazy-loaded for flexibility)
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}  # IMPORTANT for FAISS accuracy
    )


def build_faiss_index(docs: List[Document], index_name: str):
    logger.info(f"[FAISS] Building index '{index_name}' using MiniLM-L6-v2...")
    logger.info(f"[FAISS] Total documents: {len(docs)}")

    embedding_model = get_embedding_model()

    # Build index
    db = FAISS.from_documents(docs, embedding_model)

    os.makedirs(INDEX_DIR, exist_ok=True)

    # Save FAISS index + metadata
    db.save_local(INDEX_DIR, index_name=index_name)

    logger.info(f"[FAISS] Saved index → {INDEX_DIR}/{index_name}.faiss")
    logger.info(f"[FAISS] Metadata → {INDEX_DIR}/{index_name}.pkl")

    return db


def load_faiss_index(index_name: str):
    embedding_model = get_embedding_model()

    faiss_path = os.path.join(INDEX_DIR, f"{index_name}.faiss")
    pkl_path = os.path.join(INDEX_DIR, f"{index_name}.pkl")

    if not os.path.exists(faiss_path) or not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"[FAISS] Missing index files:\n"
            f" - {faiss_path}\n"
            f" - {pkl_path}\n"
            f"Index cannot be loaded."
        )

    logger.info(f"[FAISS] Loading index '{index_name}' from {INDEX_DIR}...")

    db = FAISS.load_local(
        folder_path=INDEX_DIR,
        embeddings=embedding_model,
        index_name=index_name,
        allow_dangerous_deserialization=True,
    )

    logger.info(f"[FAISS] Index loaded successfully.")
    return db
