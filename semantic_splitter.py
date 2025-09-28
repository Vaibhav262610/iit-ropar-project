from pathway import UDF
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, TypedDict
import pathway as pw
import numpy as np
import spacy
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import Document
from linked_chunks import linking

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model1 = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

class SentenceCombination(TypedDict):
    sentence: str
    index: int
    combined_sentence: str
    combined_sentence_embedding: list[float]


BreakpointThresholdType = Literal[
    "percentile", "standard_deviation", "interquartile", "gradient"
]


BREAKPOINT_DEFAULTS: dict[BreakpointThresholdType, float] = {
    "percentile": 95,
    "standard_deviation": 3,
    "interquartile": 1.5,
    "gradient": 95,
}


def split_by_spacy(text: str) -> List[str]:
    """
    Split text into sentences using spaCy.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [sent.text for sent in doc.sents]


class SemanticSplitterPathway(SemanticSplitterNodeParser):

    breakpoint_threshold_type: BreakpointThresholdType
    breakpoint_threshold_amount: Optional[float]
    sentence_splitter: Callable[[str], List[str]] = split_by_spacy
    minimum_chunk_size: int = 60

    def _get_threshold(self, distances: list[float]) -> float:
        if self.breakpoint_threshold_amount is None:
            self.breakpoint_threshold_amount = BREAKPOINT_DEFAULTS[
                self.breakpoint_threshold_type
            ]

        if self.breakpoint_threshold_type == "percentile":
            breakpoint_distance_threshold = np.percentile(
                distances, self.breakpoint_threshold_amount
            )
        elif self.breakpoint_threshold_type == "standard_deviation":
            breakpoint_distance_threshold = np.mean(distances) + (
                self.breakpoint_threshold_amount * np.std(distances)
            )
        elif self.breakpoint_threshold_type == "interquartile":
            q1, q3 = np.percentile(distances, [25, 75])
            iqr = q3 - q1

            breakpoint_distance_threshold = (
                np.mean(distances) + self.breakpoint_threshold_amount * iqr
            )
        elif self.breakpoint_threshold_type == "gradient":
            distance_gradient = np.gradient(distances, np.arange(len(distances)))
            breakpoint_distance_threshold = np.percentile(
                distance_gradient, self.breakpoint_threshold_amount
            )

        return float(breakpoint_distance_threshold)

    def _build_node_chunks(
        self, sentences: list[SentenceCombination], distances: list[float]
    ) -> list[str]:
        chunks = []
        if len(distances) > 0:
            breakpoint_distance_threshold = self._get_threshold(distances)

            indices_above_threshold = [
                i for i, x in enumerate(distances) if x > breakpoint_distance_threshold
            ]
            start_index = 0

            for index in indices_above_threshold:
                group = sentences[start_index : index + 1]
                combined_text = "".join([d["sentence"] for d in group])
                chunks.append(combined_text)

                start_index = index + 1

            if start_index < len(sentences):
                combined_text = "".join(
                    [d["sentence"] for d in sentences[start_index:]]
                )
                chunks.append(combined_text)

            # Check token count and merge with the previous chunk if necessary
            i = 1
            nlp = spacy.load("en_core_web_sm")
            while i < len(chunks):
                doc = nlp(chunks[i])
                if len(doc) < self.minimum_chunk_size:
                    # Merge with the previous chunk
                    chunks[i - 1] = chunks[i - 1] + chunks[i]
                    del chunks[i]
                else:
                    i += 1
        else:
            chunks = [" ".join([s["sentence"] for s in sentences])]

        return chunks

    def __call__(self, text: pw.ColumnExpression, **kwargs) -> pw.ColumnExpression:
        """Split given strings into smaller chunks.

        Args:
            - messages (ColumnExpression[str]): Column with texts to be split
            - **kwargs: override for defaults set in the constructor
        """
        return super().__call__(text, **kwargs)


class SemanticSplitterPathway(UDF):
    def __init__(
        self,
        breakpoint_threshold_type: BreakpointThresholdType = "percentile",
        breakpoint_threshold_amount: Optional[float] = 95,
        minimum_chunk_size: int = 60,
        sentence_splitter: Callable[[str], List[str]] = split_by_spacy,
        splitter=SemanticSplitterPathway,
        buffer_size: int = 2,
    ):
        super().__init__()
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.minimum_chunk_size = minimum_chunk_size
        self.sentence_splitter = sentence_splitter
        self.splitter = splitter(
            breakpoint_threshold_type=self.breakpoint_threshold_type,
            breakpoint_threshold_amount=self.breakpoint_threshold_amount,
            minimum_chunk_size=self.minimum_chunk_size,
            sentence_splitter=self.sentence_splitter,
            embed_model=embed_model1,
            buffer_size=buffer_size,
        )
        self.doc_num = 0

    def __wrapped__(self, txt: str, **kwargs) -> list[tuple[str, dict]]:
        doc = Document(text=txt)
        self.doc_num += 1
        chunks = self.splitter.get_nodes_from_documents([doc])
        splits = []
        for chunk in chunks:
            splits.append((chunk.get_text(), {}))
        return linking(splits,self.doc_num)
