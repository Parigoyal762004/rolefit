"""Configuration. Everything tunable lives here so nothing is buried in a call site."""

import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL", "")

# Guards the write endpoint. Unset means ingestion over HTTP is disabled, which
# is the correct default for a public deployment. Local CLI ingestion does not
# use this; it talks to Postgres directly.
INGEST_TOKEN = os.environ.get("INGEST_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Embeddings. 3-small is 1536 dims and roughly 5x cheaper than 3-large for a
# quality difference you cannot measure on a corpus this size. If you ever swap
# it, change the vector(1536) in schema.sql and re-ingest everything.
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

# Generation. Claude for the answer, because grounded summarisation with
# citations is what it is good at.
CHAT_MODEL = "claude-sonnet-5"

# Chunking. Job descriptions are short and highly structured: a paragraph of
# company blurb, then a requirements list, then benefits. 900 characters lands
# roughly one section per chunk, and the overlap stops a requirements list from
# being guillotined halfway down.
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# Retrieval. Pull 20 from each retriever, fuse, keep the top 6 for the model.
# More context is not better; it dilutes and it costs.
CANDIDATES_PER_RETRIEVER = 20
TOP_K = 6

# Reciprocal Rank Fusion constant. 60 is the value from the original paper and
# there is no good reason to tune it before you have evals saying it matters.
RRF_K = 60

# Self-correction. If the retrieved chunks are graded irrelevant, rewrite the
# query and try again, but only this many times before answering honestly that
# the corpus does not cover it.
MAX_RETRIEVAL_ATTEMPTS = 2


def require(name: str, value: str) -> str:
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in."
        )
    return value
