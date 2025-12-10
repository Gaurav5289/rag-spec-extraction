from enum import Enum
from typing import List
import re
import google.generativeai as genai
from src.utils.config import GEMINI_API_KEY
from src.utils.logger import logger


# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)


class QueryType(str, Enum):
    SPEC = "spec"        # technical specification queries
    GENERAL = "general"  # all other questions


# High-precision specification indicators
SPEC_KEYWORDS: List[str] = [
    "torque", "tighten", "tightening", "bolt", "nut", "screw",
    "clearance", "gap", "stroke", "play", "pressure", "psi", "bar",
    "capacity", "volume", "oil", "fuel", "coolant", "level",
    "dimension", "length", "width", "height", "thickness",
    "rpm", "speed", "voltage", "current", "amp", "amps", "watt",
    "power", "resistance", "ohm",
]

# Regex for numeric specifications (helps when no keyword)
SPEC_PATTERN = re.compile(r"\b\d+(\.\d+)?\s*(nm|ft|lb|psi|mm|cm|inch|in|amp|ohm|bar|kg)\b", re.IGNORECASE)


def classify_query_llm(query: str) -> QueryType:
    """
    Backup semantic classifier using Gemini when the keyword score is ambiguous.
    """
    prompt = f"""
Classify the following user query into one of two categories:

1. "spec"    → If the user is asking for any numeric or technical specification 
                (torque, pressure, gap, bolt size, capacity, value, measurement, etc.)
2. "general" → If the query is informational, conceptual or not requesting a specific spec.

Query: "{query}"

Respond ONLY with either: spec  OR  general
"""

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        text = response.text.strip().lower()
        if "spec" in text:
            return QueryType.SPEC
        return QueryType.GENERAL
    except Exception as e:
        logger.error(f"[CLASSIFIER] Gemini fallback failed: {e}")
        return QueryType.GENERAL



def classify_query(query: str) -> QueryType:
    """
    Improved multi-layer classifier:
    1. Keyword rule match
    2. Numeric/unit spec pattern
    3. LLM fallback (Gemini)
    """
    q = query.lower()

    # ------------------------------------------
    # Rule 1: Keyword-based classification
    # ------------------------------------------
    if any(keyword in q for keyword in SPEC_KEYWORDS):
        return QueryType.SPEC

    # ------------------------------------------
    # Rule 2: Regex-based numeric-unit detection
    # Example: "35 Nm", "23 psi", "3 mm", "550 ft-lb"
    # ------------------------------------------
    if SPEC_PATTERN.search(q):
        return QueryType.SPEC

    # ------------------------------------------
    # Rule 3: Fall back to Gemini semantic classification
    # ------------------------------------------
    logger.info("[CLASSIFIER] Using LLM fallback for ambiguous query.")
    return classify_query_llm(query)
