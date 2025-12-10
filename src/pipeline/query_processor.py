# src/pipeline/query_processor.py

from typing import List
from langchain_core.documents import Document

from src.parsers.parse_manager import ParseManager
from src.chunking.chunker import SpecAwareTextSplitter
from src.embeddings.embed_index import build_faiss_index, load_faiss_index
from src.retrieval.retriever import SpecRetriever
from src.retrieval.reranker import rerank_documents
from src.pipeline.query_classifier import classify_query
from src.pipeline.extraction_llm import extract_specs, SpecItem
from src.utils.logger import logger


class QueryProcessor:
    """
    Orchestrates pipeline:
    PDF → Parse → Chunk → Embed+Index → Retrieve → Rerank → Gemini extraction.
    """

    def __init__(self, index_name: str = "spec_index"):
        self.index_name = index_name
        self.vectorstore = None
        self.last_context: str = ""  # for Streamlit debug

    # --------------------------- BUILD INDEX --------------------------- #
    def build_index_from_pdf(self, file_path: str) -> None:
        logger.info(f"[QUERY] Building index from PDF: {file_path}")

        parser = ParseManager()
        docs: List[Document] = parser.load(file_path)

        splitter = SpecAwareTextSplitter()
        chunks = splitter.split_documents(docs)

        self.vectorstore = build_faiss_index(chunks, index_name=self.index_name)
        logger.info(f"[QUERY] Index '{self.index_name}' built successfully.")

    # --------------------------- LOAD INDEX ---------------------------- #
    def load_existing_index(self) -> None:
        self.vectorstore = load_faiss_index(index_name=self.index_name)
        logger.info(f"[QUERY] Index '{self.index_name}' loaded successfully.")

    # --------------------------- ANSWER QUERY -------------------------- #
    def answer_query(self, query: str) -> List[SpecItem]:
        if self.vectorstore is None:
            raise RuntimeError("Vectorstore not initialized. Build or load index first.")

        # 1. classify
        query_type = classify_query(query)
        query_type_str = query_type.value
        logger.info(f"[QUERY] Query classified as: {query_type_str}")

        # 2. retrieve
        retriever = SpecRetriever(self.vectorstore)
        initial_docs = retriever.retrieve(query, query_type=query_type_str, top_k=15)

        logger.debug("\n===== INITIAL 15 DOCUMENTS =====")
        for i, d in enumerate(initial_docs):
            logger.debug(
                f"[#{i+1}] Page={d.metadata.get('page')} "
                f"Score={d.metadata.get('score', 1.0):.3f} "
                f"Preview='{d.page_content[:120]}...'"
            )

        # 3. rerank
        ranked_docs = rerank_documents(initial_docs, query, query_type=query_type_str)
        final_docs = ranked_docs[:6]
        logger.info(f"[QUERY] Selected top {len(final_docs)} reranked chunks.")

        logger.debug("===== FINAL TOP 6 DOCUMENTS =====")
        for i, d in enumerate(final_docs):
            logger.debug(
                f"[{i+1}] Page={d.metadata.get('page')} "
                f"HybridScore={d.metadata.get('hybrid_score', 0.0):.3f} "
                f"Preview='{d.page_content[:140]}...'"
            )

        # 4. build context string + save for UI
        blocks = []
        logger.debug("\n===== CONTEXT SENT TO GEMINI =====")
        for idx, d in enumerate(final_docs, start=1):
            page = d.metadata.get("page", "?")
            section = d.metadata.get("section", "")
            score = d.metadata.get("hybrid_score", 0.0)
            text = d.page_content.strip()

            block = (
                f"\n--- CHUNK {idx} | Page {page} | Section={section} | Score={score:.2f} ---\n"
                f"{text}"
            )
            blocks.append(block)
            logger.debug(block[:600] + "...")

        context = "\n".join(blocks)
        logger.debug("===== END CONTEXT =====\n")

        # expose to Streamlit
        self.last_context = context

        # 5. LLM extraction
        specs = extract_specs(query, context, query_type=query_type_str)

        # 6. optional dedup
        unique = {}
        for s in specs:
            key = (s.component.lower(), s.value.lower(), s.unit.lower())
            unique[key] = s
        specs = list(unique.values())

        return specs
