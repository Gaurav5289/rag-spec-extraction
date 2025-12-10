# reranker.py
from typing import List
import re
from langchain_core.documents import Document


# Regex patterns to detect real numeric specifications
SPEC_NUMBER_PATTERNS = [
    r"\b\d+\s*(Nm|N·m|lb-ft|ft-lb|in-lb|psi|bar)\b",
    r"\b\d+\.\d+\s*(mm|cm|inch|in)\b",
    r"\b\d+\s*(mm|cm|inch|in)\b",
]


def contains_real_spec(text: str) -> bool:
    """Returns True if the chunk contains actual numeric specifications."""
    text = text.lower()
    for pattern in SPEC_NUMBER_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def score_document(doc: Document, query: str, query_type: str) -> float:
    """
    Hybrid scoring:
      1. FAISS semantic score (already in doc.metadata['score'])
      2. Boost if REAL torque values detected (critical)
      3. Boost for mechanical keywords
      4. Heavy boost if query tokens appear in text
    """

    text = doc.page_content.lower()
    q = query.lower()

    # 1. base semantic score from FAISS (smaller = better)
    semantic_score = float(doc.metadata.get("score", 1.0))

    # Convert FAISS score (lower is better) → normalized similarity
    semantic_sim = 1 / (1 + semantic_score)

    boost = 0.0

    # 2. HUGE boost if chunk contains REAL specs (numbers + units)
    if contains_real_spec(text):
        boost += 1.2    # ⭐ THIS FIXES YOUR PROBLEM

    # 3. Boost for domain-specific keywords
    keywords = ["torque", "bolt", "caliper", "rear", "front", "specification"]
    for k in keywords:
        if k in text:
            boost += 0.1

    # 4. Query token overlap
    for token in q.split():
        if len(token) > 3 and token in text:
            boost += 0.15

    # Limit boost
    boost = min(boost, 2.0)

    # Final score = 70% semantic + 30% boosting
    final_score = 0.7 * semantic_sim + 0.3 * boost
    return final_score


def rerank_documents(docs: List[Document], query: str, query_type: str) -> List[Document]:
    """Rerank docs by the hybrid score."""
    for d in docs:
        score = score_document(d, query, query_type)
        d.metadata["hybrid_score"] = score

    return sorted(docs, key=lambda d: d.metadata["hybrid_score"], reverse=True)
