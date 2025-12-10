# 🚗 Spec Extraction RAG (LangChain + FAISS + Gemini)

This project is a **Retrieval-Augmented Generation (RAG)** system designed to efficiently extract **vehicle specifications** from an automotive **service manual PDF**.

It leverages the following technologies:
- **LLMs:** Google Gemini for robust reasoning and structured data extraction
- **Dense Retrieval:** HuggingFace embeddings and **FAISS** (Facebook AI Similarity Search) for high-performance vector indexing
- **Interface:** A user-friendly Streamlit application

The system can answer complex, spec-oriented questions, such as:
- *"Torque for rear brake caliper bolts?"*
- *"Part number for shock absorber lower bolt?"*

It is designed to return the answers in a reliable, structured JSON format.

---

## 1. Problem & Goal

Given a voluminous service manual PDF (containing hundreds of pages, complex torque tables, detailed part lists, etc.), the RAG pipeline is engineered to:

1. **Ingestion & Indexing:** Process the PDF and index its chunks in a FAISS vector store
2. **Retrieval:** Retrieve the most relevant chunks based on a user query
3. **Extraction:** Ask a Google Gemini LLM to extract the key specifications into a rigorous **structured format**

### Target Structured Output

The LLM is prompted to extract the following fields: `component`, `value`, `unit`, `page`, `raw_text`, and `part_number`.

---

## 2. Components

This section explains how each component of the system works during Indexing and Runtime.

### 2.1 PDF Parsing

- **PyMuPDFParser** extracts text per page using `fitz`
- **LlamaParser** (optional) uses Llama Cloud for better layout-aware parsing
- **OCREngine** applies Tesseract OCR if pages come out empty (e.g., scanned diagrams)
- All of this is orchestrated by **ParseManager**

**Outcome:** You get normalized, clean text blocks ready for chunking.

### 2.2 Section-Aware Chunking

Located in `chunking/SpecAwareTextSplitter`, this component creates semantically meaningful chunks, not random splits.

- Inherits from `RecursiveCharacterTextSplitter`
- Splits into ~1000-character overlapping chunks for better context
- Detects headings using regex (e.g., `SECTION 206-03: Front Disc Brake`, ALL-CAPS headings, or known part names)
- Stores detected section names inside metadata → `section=<heading>`

**Why this matters:**
- ✔ Retrieval becomes more accurate
- ✔ The LLM receives structured sections instead of noisy slices
- ✔ Multi-part queries become easier to answer

### 2.3 Embeddings + FAISS Index

Located in `embed_index.py`, this step converts each chunk into a dense vector and stores it in FAISS.

- Uses `sentence-transformers/all-MiniLM-L6-v2` via `HuggingFaceEmbeddings`
- **`build_faiss_index(chunks, index_name)`**
  - Embeds all chunks
  - Builds a FAISS index
  - Saves index to: `data/index/<index_name>.faiss` (plus metadata)
- **`load_faiss_index(index_name)`**
  - Loads the saved FAISS index instantly (no recomputation)

**Outcome:** A high-quality semantic search engine specialized for your PDF manual.

### 2.4 Retriever (SpecRetriever)

- Calls `similarity_search_with_score(query, k)` on the FAISS store
- For each `(doc, score)`, it stores the numeric FAISS similarity in `doc.metadata["score"]`
- Returns a list of `Document` objects enriched with similarity scores

### 2.5 Hybrid Reranker

Located in `reranking.py`, computes a new `hybrid_score` for each document:

**Base:** FAISS semantic score (`doc.metadata["score"]`)

**Boosts:**
- Domain keywords in text (torque, bolt, caliper, Nm, psi, capacity, etc.)
- Overlap between query tokens and document text
- Strong extra boost if query is torque-related and chunk mentions both Nm and bolt

**Final:** `hybrid = 0.70 * semantic_score + 0.30 * boost`

Sorts documents by `hybrid_score` and keeps top-K (e.g., 6) as LLM context.

### 2.6 LLM Extraction

Located in `extraction_llm.py`, handles the final extraction step with Google Gemini.

### 2.7 Query Orchestrator (QueryProcessor)

The central class that ties everything together:

- **`build_index_from_pdf(file_path)`**
  - `ParseManager` → `SpecAwareTextSplitter` → `build_faiss_index`
  
- **`load_existing_index()`**
  - Loads FAISS index with the configured index name

- **`answer_query(query)`**:
  1. Classify query as spec or general (`QueryType`)
  2. Retrieve top-15 chunks with FAISS scores
  3. Rerank with hybrid scoring; keep top-6
  4. Build a context string from these chunks and store it as `self.last_context` for debugging in UI
  5. Pass query + context to Gemini via `extract_specs`
  6. Deduplicate final specs by `(component, value, unit)` to avoid duplicates

### 2.8 Streamlit UI

Located in `src/ui/app.py`, provides an intuitive interface for the system.

**Sidebar Features:**
- Upload PDF
- Index name text box
- "Build index from uploaded PDF" button
- "Load existing index" button

**Main Interface:**
- Query text input
- "Run Query" button
- Results table displaying: component, value, unit, page
- **"Full JSON Output" expander** - Shows complete `SpecItem` objects (including `raw_text` and `part_number`)
- **"Retrieved Chunks Sent to Gemini (Debug)" expander** - Prints `QueryProcessor.last_context`, making it clear exactly what the LLM saw

**Why the debug view matters:** It allows you to inspect the exact context chunks that were fed to the LLM, helping diagnose retrieval issues or understand why certain specs were extracted.

---

## 3. Project Structure

```
rag-spec-extraction/
│
├── src/
│   ├── parsers/
│   │   └── parse_manager.py
│   ├── chunking/
│   │   └── chunker.py
│   ├── embeddings/
│   │   └── embed_index.py
│   ├── retrieval/
│   │   ├── retriever.py
│   │   └── reranker.py
│   ├── pipeline/
│   │   ├── query_processor.py
│   │   ├── extraction_llm.py
│   │   └── query_classifier.py
│   ├── ui/
│   │   └── app.py
│   └── utils/
│       ├── config.py
│       └── logger.py
│
├── indexes/
├── raw_pdfs/
├── README.md
└── requirements.txt
```

---

## 4. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-spec-extraction.git
cd rag-spec-extraction

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your API keys (Gemini, etc.)
```

---

## 5. Usage

### Build Index
```bash
python src/embeddings/embed_index.py --pdf raw_pdfs/service_manual.pdf --index my_index
```

### Run Streamlit App
```bash
streamlit run src/ui/app.py
```

---

## 6. Features

- ✅ **Section-aware chunking** for better context preservation
- ✅ **Hybrid reranking** combining semantic + keyword matching
- ✅ **Structured JSON output** with confidence scores
- ✅ **OCR fallback** for scanned pages
- ✅ **Interactive UI** with Streamlit

---

## 8. Ideas for Improvement

Here are potential enhancements to make this RAG system even more powerful:

### 8.1 Performance & Scalability
- **Multi-index support** - Handle multiple manuals simultaneously with index switching
- **Incremental indexing** - Update only changed sections instead of rebuilding entire index
- **GPU acceleration** - Leverage CUDA for faster embedding generation on large documents
- **Caching layer** - Add Redis/LRU cache for frequently asked queries to reduce API calls

### 8.2 Retrieval Enhancements
- **Multi-query expansion** - Generate multiple query variations to improve recall
- **Parent-child retrieval** - Fetch smaller chunks but return larger parent context to LLM
- **Cross-encoder reranking** - Add a second-stage reranker (e.g., `cross-encoder/ms-marco-MiniLM`) for higher precision
- **Metadata filtering** - Enable pre-filtering by section, component type, or page ranges

### 8.3 Extraction & Output
- **Confidence scoring** - Add reliability scores for each extracted specification
- **Multi-modal support** - Extract specs from diagrams and tables using vision models
- **Batch processing** - Process multiple queries in parallel for efficiency
- **Export formats** - Support CSV, Excel, or database export of extracted specs
- **Comparison mode** - Compare specs across different model years or variants

### 8.4 User Experience
- **Query suggestions** - Auto-suggest common queries based on manual content
- **Visual feedback** - Highlight relevant sections in PDF viewer alongside answers
- **Chat history** - Maintain conversation context for follow-up questions
- **Voice input** - Allow mechanics to query hands-free using speech recognition
- **Mobile-responsive UI** - Optimize interface for tablet/mobile workshop use

### 8.5 Quality & Reliability
- **Hallucination detection** - Flag when LLM answer lacks supporting evidence in retrieved chunks
- **Human-in-the-loop** - Add review workflow for critical specs (safety torques, clearances)
- **A/B testing framework** - Compare different retrieval/prompting strategies systematically
- **Evaluation suite** - Build ground-truth Q&A dataset for automated testing
- **Logging & monitoring** - Track query patterns, failures, and retrieval quality metrics

### 8.6 Integration
- **API endpoints** - RESTful API for integration with workshop management systems
- **Database sync** - Auto-populate parts database from extracted specifications
- **Version control** - Track manual revisions and spec changes over time
- **Multi-language** - Support manuals in different languages with translation layers

### 8.7 Advanced Features
- **Contextual learning** - Fine-tune embeddings on automotive domain for better retrieval
- **Anomaly detection** - Flag unusual or conflicting specs across manual sections
- **Procedural extraction** - Extract step-by-step repair procedures, not just specs
- **Diagram parsing** - Use OCR + vision models to extract torque sequences from flowcharts

---

## 9. Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.
