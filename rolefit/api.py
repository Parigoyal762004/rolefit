"""FastAPI surface.

Local:      uvicorn rolefit.api:app --reload
On Vercel:  api/index.py re-exports `app`; see vercel.json

The deployed app is read-only. There is no write endpoint, because this is a
public URL and an open ingest route on it would let anyone insert rows and spend
the API budget. Ingest is a local CLI operation against the secret key.
"""

import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import config as cfg
from . import limits
from . import supabase as sb
from .graph import ask, build_graph

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STATE: dict = {}

ALLOWED_OUTCOMES = {"applied", "no_response", "rejected", "screen",
                    "interview", "offer"}


def _graph():
    if "graph" not in _STATE:
        _STATE["graph"] = build_graph()
    return _STATE["graph"]


app = FastAPI(title="RoleFit", version="1.0.0")

# The page and API are same-origin, so no cross-origin request is legitimate
# unless it is one of these. A wildcard would let any site on the internet spend
# this deployment's LLM budget from a visitor's browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rolefit-wine.vercel.app",
                   "https://rolefit-pari.vercel.app",
                   "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["GET", "POST"],
    allow_headers=["content-type"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    resp.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    # The page is entirely self-contained: inline style and script, no external
    # requests of any kind. So the policy can be this tight.
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'; connect-src 'self'; "
        "img-src 'self' data:; base-uri 'none'; form-action 'none'; "
        "frame-ancestors 'none'")
    return resp


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    outcome: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    attempts: int


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    """Serve the demo page from the app rather than as a static asset.

    Vercel routes every path to this function once vercel.json declares one, so
    a file at the repo root never gets served and `/` came back as FastAPI's own
    JSON 404. Reading it here removes the question, and the page stays an
    ordinary editable file rather than a Python string.
    """
    try:
        with open(os.path.join(_ROOT, "index.html"), encoding="utf-8") as fh:
            return HTMLResponse(fh.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>RoleFit</h1><p>API is up. See "
                            "<a href='/api/health'>/api/health</a>.</p>")


@app.get("/api/health")
def health() -> dict:
    checks = {}
    try:
        sb.embed_one("health check")
        checks["embeddings"] = "ok"
    except Exception as exc:
        checks["embeddings"] = f"error: {str(exc)[:120]}"
    try:
        rows = sb.rpc("rolefit_corpus", {})
        checks["database"] = "ok"
        checks["documents"] = len(rows)
    except Exception as exc:
        checks["database"] = f"error: {str(exc)[:120]}"
    checks["llm_key_present"] = bool(cfg.GROQ_API_KEY)
    checks["model"] = cfg.CHAT_MODEL
    ok = (checks.get("embeddings") == "ok" and checks.get("database") == "ok"
          and checks["llm_key_present"])
    return {"status": "ok" if ok else "degraded", **checks}


@app.get("/api/corpus")
def corpus() -> dict:
    """What is actually indexed. Drives the counter on the demo page."""
    try:
        rows = sb.rpc("rolefit_corpus", {})
        by_outcome: dict[str, int] = {}
        for r in rows:
            by_outcome[r["outcome"]] = by_outcome.get(r["outcome"], 0) + 1
        return {"by_outcome": by_outcome, "total": len(rows),
                "documents": [{"company": r["company"], "role": r["role_title"],
                               "outcome": r["outcome"]} for r in rows[:50]]}
    except Exception as exc:
        return {"by_outcome": {}, "documents": [], "total": 0,
                "error": str(exc)[:200]}


@app.post("/api/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest, request: Request) -> AskResponse:
    if req.outcome is not None and req.outcome not in ALLOWED_OUTCOMES:
        raise HTTPException(status_code=400, detail="Unknown outcome filter.")

    verdict = limits.check(limits.client_ip(request))
    if not verdict.allowed:
        raise HTTPException(status_code=429, detail=verdict.reason)

    if not cfg.GROQ_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not set on this deployment.")

    try:
        return AskResponse(**ask(_graph(), req.question, req.outcome))
    except Exception as exc:
        text = str(exc)
        # The provider's own 429 body carries the organization id and account
        # tier. Truncating it was not enough: the leak sits in the first 80
        # characters. Upstream quota errors get a message written here instead
        # of anything forwarded from the provider.
        if "429" in text or "rate_limit" in text.lower():
            raise HTTPException(
                status_code=429,
                detail="This demo has used its daily model quota. It runs on a "
                       "free tier capped at 100k tokens a day, which is roughly "
                       "twelve questions. Try again tomorrow.") from exc
        if "401" in text or "invalid_api_key" in text.lower():
            raise HTTPException(
                status_code=503,
                detail="The model provider rejected this deployment's "
                       "credentials.") from exc
        # Never surface a raw exception to a public endpoint; it leaks stack
        # frames, table names and sometimes key fragments.
        raise HTTPException(status_code=500,
                            detail="Query failed. Check /api/health.") from exc
