"""Configuration. Everything tunable lives here so nothing is buried in a call site."""

import os

from dotenv import load_dotenv

load_dotenv()

# --- Supabase -------------------------------------------------------------
# These two are publishable on purpose. A Supabase publishable key is designed
# to ship in browser bundles; it grants nothing on its own because every table
# has RLS enabled with no policies. All reads go through security-definer
# functions that expose exactly what the demo needs and nothing else.
SUPABASE_URL = os.environ.get(
    "SUPABASE_URL", "https://usfxjkroohbttyntpymd.supabase.co")
SUPABASE_PUBLISHABLE_KEY = os.environ.get(
    "SUPABASE_PUBLISHABLE_KEY", "sb_publishable_Aq5OtPNRmENil05atlZ6OQ_nSK7MIcW")

# Writes only. Never set this in the deployed environment: the public app is
# read-only by design. Ingest is a local CLI operation.
SUPABASE_SECRET_KEY = os.environ.get("SUPABASE_SECRET_KEY", "")

# --- Models ---------------------------------------------------------------
# xAI exposes no embeddings endpoint, so embeddings come from a Supabase Edge
# Function running gte-small on Supabase's own infrastructure. Free, no extra
# vendor, 384 dimensions, already normalised for cosine distance.
EMBED_DIM = 384

# Generation and grading. xAI is OpenAI wire-compatible, so the standard
# OpenAI client works against it with a different base URL.
XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
XAI_BASE_URL = os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "grok-4.5")

# --- Chunking -------------------------------------------------------------
# Job descriptions are short and highly structured: a paragraph of company
# blurb, then a requirements list, then benefits. 900 characters lands roughly
# one section per chunk, and the overlap stops a requirements list from being
# guillotined halfway down.
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# --- Retrieval ------------------------------------------------------------
# Pull 20 from each retriever, fuse, keep the top 6. More context is not
# better; it dilutes the answer and it costs tokens.
CANDIDATES_PER_RETRIEVER = 20
TOP_K = 6

# Reciprocal Rank Fusion constant. 60 is the value from the original paper.
# No reason to tune it before the evals say it matters.
RRF_K = 60

# If retrieved chunks are graded irrelevant, rewrite the query and retry, but
# only this many times before answering honestly that the corpus does not cover
# the question.
MAX_RETRIEVAL_ATTEMPTS = 2

HTTP_TIMEOUT = 30.0


def require(name: str, value: str) -> str:
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in."
        )
    return value
