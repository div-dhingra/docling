import os
import pytest
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions, VlmEngineType
from pydantic import AnyUrl


def test_e2e_cohere_conversion():
    """Test Cohere VLM conversion on a PDF file."""
    
    # Skip the test if the developer hasn't provided a Cohere API key in their environment
    if not os.environ.get("COHERE_API_KEY"):
        pytest.skip("COHERE_API_KEY is not set in environment. Skipping Cohere VLM test.")

    # Setup the converter with Cohere OCR VLM options using the new preset system
    cohere_engine_options = ApiVlmEngineOptions(
        engine_type=VlmEngineType.API,
        url=AnyUrl("https://api.cohere.ai/compatibility/v1/chat/completions"),
        headers={"Authorization": f"Bearer {os.environ.get('COHERE_API_KEY', '')}"},
        timeout=120,
        concurrency=4,
    )

    vlm_convert_options = VlmConvertOptions.from_preset(
        "cohere_ocr", engine_options=cohere_engine_options
    )

    pipeline_options = VlmPipelineOptions(
        vlm_options=vlm_convert_options,
        enable_remote_services=True, # Critical: allows the pipeline to make external API calls
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=pipeline_options,
            ),
        }
    )

    # Convert a sample PDF
    pdf_path = Path("./tests/data/pdf/2206.01062.pdf")
    conv_result = converter.convert(pdf_path)
    doc = conv_result.document

    # Basic assertions to ensure conversion populated the document structure
    assert len(doc.pages) > 0, "Document should have pages" 
    assert len(doc.texts) > 0, "Document should have text elements extracted by Cohere" 
