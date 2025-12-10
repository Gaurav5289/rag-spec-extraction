import pytesseract
import fitz
from PIL import Image
from typing import List
from langchain_core.documents import Document
from src.utils.logger import logger
import re


class OCREngine:
    """
    OCR fallback for pages with missing or extremely short text.
    Extracts meaningful OCR text and ensures correct metadata.
    """

    MIN_TEXT_CHARS = 25  # If fewer chars → treat page as empty/needs OCR

    def ocr_pages_if_empty(self, file_path: str, docs: List[Document]) -> List[Document]:
        logger.info(f"[OCR] Starting OCR fallback for file: {file_path}")

        try:
            pdf = fitz.open(file_path)
        except Exception as e:
            logger.error(f"[OCR] Could not open PDF → {e}")
            return docs

        updated_docs = []

        for idx, doc in enumerate(docs):
            page_num = doc.metadata.get("page", idx + 1)
            text = (doc.page_content or "").strip()

            # Skip OCR if page already has enough real text
            if len(text) >= self.MIN_TEXT_CHARS:
                updated_docs.append(doc)
                continue

            logger.warning(f"[OCR] Page {page_num} seems empty or very short (chars={len(text)}). Running OCR.")

            try:
                page = pdf.load_page(idx)

                # High-resolution pixmap improves OCR accuracy
                pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                ocr_text_raw = pytesseract.image_to_string(img)

                # Clean OCR text
                clean_text = re.sub(r"[ \t]+", " ", ocr_text_raw)
                clean_text = re.sub(r"\n{2,}", "\n", clean_text).strip()

                if len(clean_text) < 5:
                    logger.warning(f"[OCR] Page {page_num} produced almost no OCR text; keeping original.")
                    updated_docs.append(doc)
                    continue

                # Create a proper document with metadata preserved
                new_doc = Document(
                    page_content=clean_text,
                    metadata={
                        "page": page_num,
                        "section": doc.metadata.get("section", ""),  # Keep original if exists
                        "source": file_path,
                        "file_name": file_path.split("/")[-1],
                        "parser": "ocr",
                        "ocr": True,
                    },
                )

                logger.info(f"[OCR] Page {page_num}: OCR extracted {len(clean_text)} chars.")
                updated_docs.append(new_doc)

            except pytesseract.TesseractNotFoundError:
                logger.error(
                    f"[OCR] Tesseract not installed. Install Tesseract and set PATH. "
                    f"Skipping OCR for page {page_num}."
                )
                updated_docs.append(doc)

            except Exception as e:
                logger.error(f"[OCR] Unexpected error on page {page_num} → {e}")
                updated_docs.append(doc)

        pdf.close()
        logger.info("[OCR] OCR fallback process complete.")
        return updated_docs
