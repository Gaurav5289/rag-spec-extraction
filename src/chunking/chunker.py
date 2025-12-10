from typing import List, Any
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.utils.logger import logger

# Robust detection for REAL section titles
SECTION_PATTERN = re.compile(
    r"(SECTION\s+\d{3}(?:[-A-Z0-9]+)?(?:\s*:\s*[A-Za-z0-9 \-]+)?)",
    re.IGNORECASE,
)

class SpecAwareTextSplitter(RecursiveCharacterTextSplitter):
    """
    Improved chunker for service manuals.
    - Preserves section titles
    - Reduces chunk size for higher recall
    - Adds chunk metadata
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            chunk_size=600,          # MUCH better for Gemini 2.5
            chunk_overlap=80,        # Enough context without duplication
            separators=["\n\n", "\n", ". ", " ", ""],
            **kwargs,
        )

    def split_documents(self, docs: List[Document]) -> List[Document]:
        output: List[Document] = []
        logger.info(f"[Chunker] Splitting {len(docs)} parsed pages...")

        for doc_index, doc in enumerate(docs):
            page_text = doc.page_content

            # Keep original section from parser
            original_section = doc.metadata.get("section", "")
            page_num = doc.metadata.get("page", "?")

            chunks = super().split_text(page_text)

            for chunk_index, chunk in enumerate(chunks):

                metadata = dict(doc.metadata)
                metadata["chunk_index"] = chunk_index
                metadata["parent_page"] = page_num
                metadata["original_section"] = original_section

                # Attempt to detect section header *only if missing*
                if not original_section:
                    match = SECTION_PATTERN.search(chunk)
                    if match:
                        metadata["section"] = match.group(1).strip()

                # Clean chunk
                cleaned_chunk = chunk.strip()

                output.append(
                    Document(
                        page_content=cleaned_chunk,
                        metadata=metadata
                    )
                )

        logger.info(f"[Chunker] Produced {len(output)} total chunks.")
        return output
