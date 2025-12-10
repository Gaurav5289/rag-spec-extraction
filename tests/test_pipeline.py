import os
import pytest

from src.pipeline.query_processor import QueryProcessor
from src.utils.config import RAW_DIR


@pytest.mark.skip(reason="Integration test, requires sample PDF & API keys.")
def test_build_and_query():
    sample_pdf = os.path.join(RAW_DIR, "sample-service-manual.pdf")
    assert os.path.exists(sample_pdf), "Put a test PDF in data/raw first."

    qp = QueryProcessor(index_name="test_index")
    qp.build_index_from_pdf(sample_pdf)

    specs = qp.answer_query("Torque for rear brake caliper bolts")
    assert isinstance(specs, list)
