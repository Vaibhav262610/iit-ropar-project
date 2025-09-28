import logging
from pathway.xpacks.llm.embedders import OpenAIEmbedder, SentenceTransformerEmbedder
from pathway.xpacks.llm.splitters import TokenCountSplitter
from pathway.xpacks.llm.vector_store import VectorStoreServer
from pathway.xpacks.llm.parsers import ParseUnstructured, OpenParse
import sys
from splade_embed import SpladeEmbedder
from semantic_splitter import SemanticSplitterPathway

logging.basicConfig(stream=sys.stderr, level=logging.WARN, force=True)

import pathway as pw

data_sources = []
data_sources.append(
    pw.io.fs.read(
        "./dev",
        format="binary",
        mode="streaming",
        with_metadata=True,
    )
)

PATHWAY_PORT = 8765
PATHWAY_HOST = "127.0.0.1"


# embedder = SpladeEmbedder(model="naver/splade-cocondenser-ensembledistil")
text_splitter = TokenCountSplitter(min_tokens=1000, max_tokens=1500)
embedder = SentenceTransformerEmbedder(model="infgrad/stella-base-en-v2")

parser = ParseUnstructured()

vector_server = VectorStoreServer(
    *data_sources,
    parser=parser,
    embedder=embedder,
    splitter=text_splitter,
)
print("Starting server")
vector_server.run_server(
    host=PATHWAY_HOST, port=PATHWAY_PORT, threaded=True, with_cache=False
)
