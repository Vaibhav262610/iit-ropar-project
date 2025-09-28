import json
import os
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
# read all the .pdf files in dev folder ./dev/*.pdf
paths = list(Path("./dev").glob("*.pdf"))

doc_converter = DocumentConverter()
conv_results = doc_converter.convert_all(paths)
i = 1
# store the results in results ./results folder as markdown files 1.md 2.md
for result in conv_results:
    text = result.document.export_to_markdown()
    # write to the file and create it if it does not exist
    os.makedirs("./results", exist_ok=True)
    with open(f"./results/{i}.md", "w") as f:
        f.write(text)
    i += 1
