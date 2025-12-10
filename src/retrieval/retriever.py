from typing import List, Tuple
from langchain_core.documents import Document
from src.utils.logger import logger
import re


# Keywords that should boost retrieval for SPEC-type queries
SPEC_KEYWORDS = [
    "torque", "tighten", "nut", "bolt", "caliper", "hose",
    "psi", "capacity", "fluid", "oil", "grease",
    "disc", "pad", "brake", "shock", "damper",
]


class SpecRetriever:
    """
    Hybrid retriever:
    - FAISS vector similarity
    - Lightweight keyword scoring
    - Normalized similarity values
    """

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    # --------------------------------------------------------
    # Compute keyword score
    # --------------------------------------------------------
    def keyword_score(self, text: str, query: str) -> float:
        t = text.lower()
        q = query.lower()

        score = 0.0
        for kw in SPEC_KEYWORDS:
            if kw in t:
                score += 0.3
            if kw in q and kw in t:
                score += 0.5

        # Token overlap
        for token in q.split():
            if len(token) > 3 and token in t:
                score += 0.1

        return min(score, 2.0)

    # --------------------------------------------------------
    # Main retrieval
    # --------------------------------------------------------
    def retrieve(self, query: str, query_type: str, top_k: int = 15) -> List[Document]:
        logger.info(f"[Retriever] Retrieving top-{top_k} candidates for '{query_type}' query...")

        try:
            results: List[Tuple[Document, float]] = \
                self.vectorstore.similarity_search_with_score(query, k=top_k)
        except Exception as e:
            logger.error(f"[Retriever] FAISS retrieval failed: {e}")
            return []

        docs: List[Document] = []

        # Normalize FAISS distance → similarity
        all_scores = [float(s) for _, s in results]
        max_s = max(all_scores) if all_scores else 1
        min_s = min(all_scores) if all_scores else 0
        range_s = (max_s - min_s) or 1

        for doc, distance in results:
            distance = float(distance)

            # Convert FAISS distance → similarity (lower dist = better)
            faiss_sim = 1 - ((distance - min_s) / range_s)

            # Add lexical keyword score
            kw = self.keyword_score(doc.page_content, query)

            # Hybrid score stored for reranker
            hybrid_raw = faiss_sim + kw

            doc.metadata["faiss_distance"] = distance
            doc.metadata["faiss_similarity"] = float(faiss_sim)
            doc.metadata["keyword_score"] = float(kw)
            doc.metadata["retrieval_score"] = float(hybrid_raw)
            doc.metadata["retrieval_type"] = "hybrid"

            docs.append(doc)

        return docs
