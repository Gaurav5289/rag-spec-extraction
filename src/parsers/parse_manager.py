from typing import List
from langchain_core.documents import Document

from src.parsers.llama_parser import LlamaParser
from src.parsers.pymupdf_parser import PyMuPDFParser
from src.ocr.ocr_engine import OCREngine
from src.utils.logger import logger


class ParseManager:
    """
    PDF Parsing Orchestrator:
    1. Try LlamaParse (best structured output)
    2. Fallback → PyMuPDF text extraction
    3. OCR fallback for pages with no text
    """

    def __init__(self) -> None:
        self.llama = LlamaParser()
        self.pymupdf = PyMuPDFParser()
        self.ocr = OCREngine()

    def load(self, file_path: str) -> List[Document]:
        docs: List[Document] = []

        # -----------------------------
        # 1️⃣ Try LlamaParse
        # -----------------------------
        if self.llama.client is not None:
            try:
                docs = self.llama.load(file_path)

                if docs and len(docs) > 0:
                    logger.info(f"[ParseManager] Parsed PDF using LlamaParse ({len(docs)} pages).")
                else:
                    raise ValueError("LlamaParse returned empty docs.")

            except Exception as e:
                logger.warning(f"[ParseManager] LlamaParse failed → {e}.")
                logger.warning("[ParseManager] Falling back to PyMuPDF.")
                docs = self.pymupdf.load(file_path)

        else:
            logger.info("[ParseManager] No LlamaParse client → using PyMuPDF only.")
            docs = self.pymupdf.load(file_path)

        # -----------------------------
        # 2️⃣ OCR fallback
        # -----------------------------
        docs = self.ocr.ocr_pages_if_empty(file_path, docs)

        # -----------------------------
        # 3️⃣ Fix Missing Metadata
        # -----------------------------
        for doc in docs:
            doc.metadata.setdefault("page", -1)
            doc.metadata.setdefault("section", "")
            doc.metadata.setdefault("source", file_path)
            doc.metadata.setdefault("parser", "llama" if self.llama.client else "pymupdf")

        logger.info(f"[ParseManager] Final document count: {len(docs)}")

        return docs
