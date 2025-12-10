import fitz
from typing import List
from langchain_core.documents import Document
from src.utils.logger import logger
import re


class PyMuPDFParser:
    """
    Improved PyMuPDF parser:
    - Extracts text + preserves basic layout
    - Extracts section headers
    - Cleans whitespace
    - Adds richer metadata
    """

    SECTION_REGEX = re.compile(r"(SECTION\s+\d{3}[-\w]*[:\s].*)", re.IGNORECASE)

    def load(self, file_path: str) -> List[Document]:
        logger.info(f"[PyMuPDFParser] Parsing PDF with PyMuPDF: {file_path}")

        docs = []
        pdf = fitz.open(file_path)

        try:
            for i, page in enumerate(pdf):
                raw_text = page.get_text("text")

                # Normalize whitespace
                text = re.sub(r"[ \t]+", " ", raw_text)
                text = re.sub(r"\n{2,}", "\n", text).strip()

                # Extract section header (if present)
                section_match = self.SECTION_REGEX.search(text)
                section = section_match.group(1).strip() if section_match else ""

                # Build document
                doc = Document(
                    page_content=text,
                    metadata={
                        "page": i + 1,
                        "section": section,
                        "source": file_path,
                        "file_name": file_path.split("/")[-1],
                        "parser": "pymupdf",
                    },
                )

                docs.append(doc)

        except Exception as e:
            logger.error(f"[PyMuPDFParser] Error parsing PDF: {e}")
        finally:
            pdf.close()

        logger.info(f"[PyMuPDFParser] Extracted {len(docs)} pages.")
        return docs
