"""Configuration. Everything tunable lives here so nothing is buried in a call site."""

import os

from dotenv import load_dotenv

# Load THIS project's .env and nothing else.
#
# Bare load_dotenv() walks up the directory tree looking for a .env, which means
# a file belonging to a parent folder or an unrelated project can configure this
# one. Pinning the path removes that.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_REPO, ".env"), override=True)


def _own(name: str, default: str = "") -> str:
    """Read a ROLEFIT_-prefixed variable, never a bare ambient one.

    This machine has SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY set as
    machine-wide Windows environment variables, pointing at a different
    project's production database with an admin key. Reading the unprefixed
    names would silently point this app at that database, and the failure would
    look like an auth error rather than what it is. Prefixing means ambient
    variables cannot reach this app at all.
    """
    return os.environ.get(f"ROLEFIT_{name}", default)

# --- Supabase -------------------------------------------------------------
# These two are publishable on purpose. A Supabase publishable key is designed
# to ship in browser bundles; it grants nothing on its own because every table
# has RLS enabled with no policies. All reads go through security-definer
# functions that expose exactly what the demo needs and nothing else.
SUPABASE_URL = _own("SUPABASE_URL", "https://usfxjkroohbttyntpymd.supabase.co")
SUPABASE_PUBLISHABLE_KEY = _own(
    "SUPABASE_PUBLISHABLE_KEY", "sb_publishable_Aq5OtPNRmENil05atlZ6OQ_nSK7MIcW")

# Writes only. Never set this in the deployed environment: the public app is
# read-only by design. Ingest is a local CLI operation.
SUPABASE_SECRET_KEY = _own("SUPABASE_SECRET_KEY", "")

# --- Models ---------------------------------------------------------------
# Groq exposes no embeddings endpoint, so embeddings come from a Supabase Edge
# Function running gte-small on Supabase's own infrastructure. Free, no extra
# vendor, 384 dimensions, already normalised for cosine distance.
EMBED_DIM = 384

# Generation and grading. Groq is OpenAI wire-compatible, so the standard
# OpenAI client works against it with a different base URL.
GROQ_API_KEY = _own("GROQ_API_KEY", "")
GROQ_BASE_URL = _own("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
CHAT_MODEL = _own("CHAT_MODEL", "llama-3.3-70b-versatile")

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
