from typing import List
from langchain_core.documents import Document
from llama_parse import LlamaParse
from src.utils.logger import logger
from src.utils.config import LLAMA_CLOUD_API_KEY
import re


class LlamaParser:
    """
    LlamaParse wrapper for structured PDF parsing.
    - Handles new API format
    - Splits into page documents
    - Extracts section titles
    """

    SECTION_REGEX = re.compile(r"(SECTION\s+\d{3}[-\w]*[:\s].*)", re.IGNORECASE)

    def __init__(self):
        if not LLAMA_CLOUD_API_KEY:
            self.client = None
            logger.warning("LLAMA_CLOUD_API_KEY missing → LlamaParse disabled.")
            return

        try:
            self.client = LlamaParse(api_key=LLAMA_CLOUD_API_KEY)
            logger.info("[LlamaParser] Client initialized.")
        except Exception as e:
            self.client = None
            logger.error(f"[LlamaParser] Initialization error → {e}")

    def load(self, file_path: str) -> List[Document]:
        if self.client is None:
            raise RuntimeError("LlamaParse client not initialized.")

        logger.info(f"[LlamaParser] Parsing PDF with LlamaParse: {file_path}")

        # Parse PDF
        try:
            results = self.client.load_data(file_path)
        except Exception as e:
            logger.error(f"[LlamaParser] Parsing failed → {e}")
            raise

        docs: List[Document] = []

        # LlamaParse output can be:
        # - A list of page-like documents
        # - A single giant document (must split manually)
        for idx, parsed in enumerate(results):
            page_text = parsed.page_content or ""

            # Clean text (remove excessive whitespace)
            clean = re.sub(r"[ \t]+", " ", page_text)
            clean = re.sub(r"\n{2,}", "\n", clean).strip()

            # Extract section header if present
            section_match = self.SECTION_REGEX.search(clean)
            section = section_match.group(1).strip() if section_match else ""

            # Safe page number fallback
            page_num = parsed.metadata.get("page_number") or idx + 1

            doc = Document(
                page_content=clean,
                metadata={
                    "page": page_num,
                    "section": section,
                    "source": file_path,
                    "file_name": file_path.split("/")[-1],
                    "parser": "llamaparse",
                },
            )

            docs.append(doc)

        logger.info(f"[LlamaParser] Extracted {len(docs)} page documents.")
        return docs
