import sys, os
import pandas as pd
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
from src.pipeline.query_processor import QueryProcessor
from src.utils.config import RAW_DIR
from src.utils.logger import logger
from src.pipeline.extraction_llm import SpecItem


# ==========================================================
# SESSION STATE INITIALIZATION (RUN ONLY ONCE)
# ==========================================================
if "qp" not in st.session_state:
    st.session_state.qp = None

if "index_name" not in st.session_state:
    st.session_state.index_name = "spec_index"

if "chunks_debug" not in st.session_state:
    st.session_state.chunks_debug = ""


# ==========================================================
# DISPLAY TABLE
# ==========================================================
def display_specs_table(specs: List[SpecItem]):
    df = pd.DataFrame([s.dict() for s in specs])
    df_display = df[['component', 'value', 'unit', 'page']].fillna("")
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    with st.expander("Full JSON Output"):
        st.json(df.to_dict(orient="records"))


# ==========================================================
# MAIN UI
# ==========================================================
def main():
    st.set_page_config(page_title="Spec Extraction RAG", layout="wide")
    st.title("🔧 Service Manual Spec Extraction (LangChain + Gemini)")

    st.sidebar.header("PDF & Index")

    uploaded_pdf = st.sidebar.file_uploader("Upload service manual PDF", type=["pdf"])

    # Index name input
    new_index_name = st.sidebar.text_input(
        "Index name",
        value=st.session_state.index_name,
        key="index_name_input"
    )

    # Update index name
    if new_index_name != st.session_state.index_name:
        st.session_state.index_name = new_index_name

        # RESET QueryProcessor ONLY when name changes
        st.session_state.qp = QueryProcessor(index_name=new_index_name)

    # Ensure qp exists
    if st.session_state.qp is None:
        st.session_state.qp = QueryProcessor(index_name=st.session_state.index_name)

    qp: QueryProcessor = st.session_state.qp

    # ========= LOAD EXISTING INDEX ==================================
    if st.sidebar.button("Load existing index"):
        try:
            qp.load_existing_index()

            # Persist loaded qp back to session_state  
            st.session_state.qp = qp

            st.sidebar.success(f"Loaded index '{qp.index_name}' successfully!")
        except FileNotFoundError:
            st.sidebar.error(f"Index not found: {qp.index_name}")
        except Exception as e:
            st.sidebar.error(f"Failed to load index: {e}")

    # ========= BUILD INDEX ==================================
    if st.sidebar.button("Build index from uploaded PDF"):
        if uploaded_pdf is None:
            st.sidebar.error("Please upload a PDF first.")
        else:
            os.makedirs(RAW_DIR, exist_ok=True)
            file_path = os.path.join(RAW_DIR, uploaded_pdf.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_pdf.getbuffer())

            try:
                with st.spinner(f"Building index '{qp.index_name}' ..."):
                    qp.build_index_from_pdf(file_path)

                # Persist updated qp
                st.session_state.qp = qp

                st.sidebar.success(f"Index '{qp.index_name}' built successfully!")
            except Exception as e:
                logger.error(e)
                st.sidebar.error(f"Error: {e}")

    st.header("🔍 Query Specifications")
    query = st.text_input("Enter query (e.g., 'Torque for rear brake caliper bolts')")

    # ==========================================================
    # RUN QUERY
    # ==========================================================
    if st.button("Run Query"):
        if qp.vectorstore is None:
            st.error("No index loaded. Build or load an index first.")
        elif not query.strip():
            st.error("Enter a valid query.")
        else:
            with st.spinner("Running retrieval + extraction..."):
                try:
                    specs = qp.answer_query(query)

                    # Save debug chunks
                    st.session_state.chunks_debug = getattr(qp, "last_context", "")

                    if not specs:
                        st.warning("No specifications extracted.")
                    else:
                        st.success(f"Found {len(specs)} specs.")
                        display_specs_table(specs)

                except Exception as e:
                    st.error(f"Query error: {e}")

    # ==========================================================
    # SHOW RETRIEVED CHUNKS SENT TO GEMINI
    # ==========================================================
    st.subheader("📄 Retrieved Chunks Sent to Gemini (Debug)")
    with st.expander("Show Chunks"):
        st.text(st.session_state.chunks_debug or "No chunks available.")


# ENTRY POINT
if __name__ == "__main__":
    main()
