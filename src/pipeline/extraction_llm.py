# src/pipeline/extraction_llm.py

from typing import List
import json
import re

import google.generativeai as genai
from pydantic import BaseModel, Field, ValidationError

from src.utils.logger import logger
from src.utils.config import GEMINI_API_KEY


# --------------------------------------------------
# 1. Configure Gemini
# --------------------------------------------------
genai.configure(api_key=GEMINI_API_KEY)

GEMINI_MODEL = "gemini-2.5-flash"


# --------------------------------------------------
# 2. Pydantic Schemas
# --------------------------------------------------
class SpecItem(BaseModel):
    component: str = Field(description="Component name, e.g. 'Rear brake caliper bolt'")
    value: str = Field(description="Numeric or descriptive value, e.g. '35'")
    unit: str = Field(default="", description="Unit of measurement, e.g. 'Nm'")
    page: int | None = Field(default=None, description="Page number in the manual")
    raw_text: str | None = Field(default=None, description="Original text from which this spec was extracted")


class SpecList(BaseModel):
    specs: List[SpecItem]


# --------------------------------------------------
# 3. Small helper to pull JSON out of messy output
# --------------------------------------------------
def _extract_json_block(raw: str) -> str | None:
    """Try to extract a JSON object or array from raw LLM text."""
    # First try a JSON object {...}
    obj_match = re.search(r"\{[\s\S]*\}", raw)
    if obj_match:
        return obj_match.group(0)

    # Then try a top-level array [...]
    arr_match = re.search(r"\[[\s\S]*\]", raw)
    if arr_match:
        return arr_match.group(0)

    return None


# --------------------------------------------------
# 4. Main extraction function (Option 1: Safe JSON)
# --------------------------------------------------
def extract_specs(query: str, context: str, query_type: str) -> List[SpecItem]:
    """
    Use Gemini to extract structured specifications.

    Strategy:
      1. Ask Gemini to answer ONLY in JSON.
      2. Try to parse directly with Pydantic (SpecList).
      3. If that fails, regex out the JSON block and parse again.
      4. If *that* fails, log and return [].
    """

    prompt = f"""
You are a technical specification extraction engine.

### TASK
From the context below, extract ALL technical specs that answer the user's query.

### QUERY TYPE
{query_type}

### USER QUERY
{query}

### CONTEXT (service manual excerpts)
{context}

### OUTPUT FORMAT (VERY IMPORTANT)
Return ONLY JSON and nothing else.

The JSON must match this schema:

{{
  "specs": [
    {{
      "component": "string, e.g. 'Rear brake caliper bolt'",
      "value": "string, numeric as string, e.g. '35'",
      "unit": "string, e.g. 'Nm', may be empty string if no unit",
      "page": 123,  // integer page number if known, otherwise null
      "raw_text": "original short phrase/snippet from the manual for this spec"
    }}
  ]
}}

Rules:
- Do NOT include comments in the JSON.
- Do NOT wrap the JSON in backticks.
- If no relevant specs exist, return: {{"specs": []}}
"""

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.0,
                "candidate_count": 1,
                "max_output_tokens": 2048,
            },
        )

        raw_text = (response.text or "").strip()
        logger.debug("[LLM] Raw output (first 500 chars):\n" + raw_text[:500])

        # 1. Try direct JSON → SpecList
        try:
            data = SpecList.model_validate_json(raw_text)
            logger.info(f"[LLM] Parsed JSON directly. Specs: {len(data.specs)}")
            return data.specs
        except ValidationError:
            logger.warning("[LLM] Direct JSON validation failed. Trying regex extraction...")

        # 2. Regex out JSON block
        json_block = _extract_json_block(raw_text)
        if not json_block:
            logger.warning("[LLM] Could not find JSON block in model output.")
            return []

        # Sometimes model returns just an array [...], wrap into {"specs": [...]} if needed
        stripped = json_block.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            json_block = json.dumps({"specs": json.loads(stripped)})

        data = SpecList.model_validate_json(json_block)
        logger.info(f"[LLM] Parsed JSON after regex recovery. Specs: {len(data.specs)}")
        return data.specs

    except Exception as e:
        logger.error(f"[LLM] Extraction failed: {e}")
        return []
