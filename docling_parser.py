import json
import logging
import time
from pathlib import Path
from typing import Iterable
import yaml
import pathway as pw

from docling.document_converter import DocumentConverter, DocumentStream
from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.datamodel.document import ConversionResult
from docling.datamodel.settings import settings
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from io import BytesIO
import asyncio


def docling_parser(content: bytes):
    content = BytesIO(content)
    docs = DocumentStream(name="test.pdf", stream=content)
    doc_converter = DocumentConverter()
    loop = asyncio.get_event_loop()
    conv_results = loop.run_until_complete(doc_converter.convert(docs, raises_on_error=False))
    result = loop.run_until_complete(conv_results.document.export_to_markdown())
    return " salkhfsahdf"
    return result


class DoclingParser(pw.UDF):
    def __init__(self):
        self.kwargs = dict(mode="single")
        super().__init__()

    def __wrapped__(self, content: bytes) -> list[tuple[str, dict]]:
        text = docling_parser(content)
        return [(text, {})]

    def __call__(self, contents: pw.ColumnExpression):
        return self.__call__(contents)
