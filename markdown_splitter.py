import re
import os
from typing import Any, Dict, List, Tuple, TypedDict, Union
from langchain_core.documents import Document
from langchain_text_splitters.base import Language
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
import pathway as pw


class ExperimentalMarkdownSyntaxTextSplitter:
    DEFAULT_HEADER_KEYS = {
        "#": "Header 1",
        "##": "Header 2",
        "###": "Header 3",
        "####": "Header 4",
        "#####": "Header 5",
        "######": "Header 6",
    }

    def __init__(
        self,
        headers_to_split_on: Union[List[Tuple[str, str]], None] = None,
        return_each_line: bool = False,
        strip_headers: bool = True,
        convert_tables: bool = True,
    ):
        self.chunks: List[Document] = []
        self.current_chunk = Document(page_content="")
        self.current_header_stack: List[Tuple[int, str]] = []
        self.convert_tables = convert_tables
        self.strip_headers = strip_headers
        if headers_to_split_on:
            self.splittable_headers = dict(headers_to_split_on)
        else:
            self.splittable_headers = self.DEFAULT_HEADER_KEYS

        self.return_each_line = return_each_line

    def transform_documents(self, text: List[Document]):
        final = []
        for d in text:
            self.chunks = self.split(d.page_content)
            text_splitter = SemanticChunker(
                embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
            )

            for chunk in self.chunks:
                if not chunk.page_content.strip():
                    continue
                split_content = text_splitter.split_text(chunk.page_content)
                for i in range(len(split_content)):
                    if split_content[i] and len(split_content[i]) > 0:
                        final.append(
                            Document(
                                page_content=split_content[i], metadata=chunk.metadata
                            )
                        )

        return final

    def split(self, text: str) -> List[Document]:
        raw_lines = text.splitlines(keepends=True)

        while raw_lines:
            raw_line = raw_lines.pop(0)
            header_match = self._match_header(raw_line)
            code_match = self._match_code(raw_line)
            horz_match = self._match_horz(raw_line)
            table_match = self._match_table(raw_line)
            if header_match:
                self._complete_chunk_doc()

                if not self.strip_headers:
                    self.current_chunk.page_content += raw_line

                # add the header to the stack
                header_depth = len(header_match.group(1))
                header_text = header_match.group(2)
                self._resolve_header_stack(header_depth, header_text)
            elif code_match:
                self._complete_chunk_doc()
                self.current_chunk.page_content = self._resolve_code_chunk(
                    raw_line, raw_lines
                )
                self.current_chunk.metadata["Code"] = code_match.group(1)
                self._complete_chunk_doc()
            elif horz_match:
                self._complete_chunk_doc()
            elif table_match:
                self._complete_chunk_doc()
                self.current_chunk = self._resolve_table_chunk(raw_line, raw_lines)
                self._complete_chunk_doc()
            else:
                self.current_chunk.page_content += raw_line

        self._complete_chunk_doc()
        if self.return_each_line:
            return [
                Document(page_content=line, metadata=chunk.metadata)
                for chunk in self.chunks
                for line in chunk.page_content.splitlines()
                if line and not line.isspace()
            ]
        return self.chunks

    def _resolve_header_stack(self, header_depth: int, header_text: str) -> None:
        for i, (depth, _) in enumerate(self.current_header_stack):
            if depth == header_depth:
                self.current_header_stack[i] = (header_depth, header_text)
                self.current_header_stack = self.current_header_stack[: i + 1]
                return
        self.current_header_stack.append((header_depth, header_text))

    def _resolve_code_chunk(self, current_line: str, raw_lines: List[str]) -> str:
        chunk = current_line
        while raw_lines:
            raw_line = raw_lines.pop(0)
            chunk += raw_line
            if self._match_code(raw_line):
                return chunk
        return ""

    def _resolve_table_chunk(self, current_line: str, raw_lines: List[str]) -> Document:
        """Process a markdown table and return it as a Document."""
        table_lines = [current_line.rstrip()]

        # Collect all table lines
        while raw_lines and self._match_table(raw_lines[0]):
            table_lines.append(raw_lines.pop(0).rstrip())

        # Parse table components
        if len(table_lines) < 3:  # Need header, separator, and at least one row
            return Document(page_content="".join(table_lines))

        # Extract headers and clean them
        headers = [cell.strip() for cell in table_lines[0].split("|")[1:-1]]

        # Skip separator line
        rows = []
        for line in table_lines[2:]:
            if line.strip():
                row = [cell.strip() for cell in line.split("|")[1:-1]]
                rows.append(row)

        # Create document with appropriate content and metadata
        if self.convert_tables:
            # Convert to natural language format
            content_lines = []
            for row in rows:
                row_texts = []
                for header, value in zip(headers, row):
                    row_texts.append(f"{header} is {value}")
                content_lines.append(". ".join(row_texts))
            content = "\n".join(content_lines)
        else:
            content = "".join(table_lines)

        # Create document with table metadata
        doc = Document(
            page_content=content,
            metadata={
                "content_type": "table",
                "table_data": {
                    "headers": headers,
                    "rows": rows,
                },
            },
        )

        for depth, value in self.current_header_stack:
            header_key = self.splittable_headers.get("#" * depth)
            doc.metadata[header_key] = value

        return doc

    def _complete_chunk_doc(self) -> None:
        chunk_content = self.current_chunk.page_content
        if chunk_content and not chunk_content.isspace():
            for depth, value in self.current_header_stack:
                header_key = self.splittable_headers.get("#" * depth)
                self.current_chunk.metadata[header_key] = value
            self.chunks.append(self.current_chunk)
        self.current_chunk = Document(page_content="")

    # Match methods
    def _match_header(self, line: str) -> Union[re.Match, None]:
        match = re.match(r"^(#{1,6}) (.*)", line)
        if match and match.group(1) in self.splittable_headers:
            return match
        return None

    def _match_code(self, line: str) -> Union[re.Match, None]:
        matches = [re.match(rule, line) for rule in [r"^```(.*)", r"^~~~(.*)"]]
        return next((match for match in matches if match), None)

    def _match_horz(self, line: str) -> Union[re.Match, None]:
        matches = [
            re.match(rule, line) for rule in [r"^\*\*\*+\n", r"^---+\n", r"^___+\n"]
        ]
        return next((match for match in matches if match), None)

    def _match_table(self, line: str) -> Union[re.Match, None]:
        """Match a table line in markdown format."""
        return re.match(r"^\s*\|.*\|\s*$", line)


class MarkdownSplitter(pw.UDF):
    def __init__(self):
        super().__init__()
        self.splitter = ExperimentalMarkdownSyntaxTextSplitter()

    def __wrapped__(self, txt: str, **kwargs) -> list[tuple[str, dict]]:
        doc = Document(page_content=txt)
        chunks = self.splitter.transform_documents([doc])
        splits = []
        for chunk in chunks:
            splits.append((chunk.page_content, {}))
        return splits
