"""FastAPI surface.

Local:      uvicorn rolefit.api:app --reload
On Vercel:  api/index.py re-exports `app`; see vercel.json

Connection handling is written for serverless, not for a long-lived server. A
Lambda that has been frozen and thawed comes back with a Postgres socket that
looks open and is not, so every request checks and reconnects rather than
trusting a module-level global. The Supabase transaction pooler on 6543 is the
right endpoint for this; the direct 5432 connection will exhaust its pool once
more than a handful of function instances are warm.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from openai import OpenAI
from pydantic import BaseModel, Field

from . import config as cfg
from .graph import ask, build_graph
from .ingest import connect, ingest_one

_STATE: dict = {}


def _oai() -> OpenAI:
    if "oai" not in _STATE:
        _STATE["oai"] = OpenAI(
            api_key=cfg.require("OPENAI_API_KEY", cfg.OPENAI_API_KEY))
    return _STATE["oai"]


def _conn():
    conn = _STATE.get("conn")
    if conn is None or conn.closed:
        conn = connect()
        _STATE["conn"] = conn
        # The graph closes over the connection, so it has to be rebuilt too.
        _STATE.pop("graph", None)
    return conn


def _graph():
    conn = _conn()
    if "graph" not in _STATE:
        _STATE["graph"] = build_graph(conn, _oai())
    return _STATE["graph"]


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    conn = _STATE.get("conn")
    if conn is not None and not conn.closed:
        conn.close()


app = FastAPI(title="RoleFit", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    outcome: str | None = Field(
        default=None,
        description="Filter to one application result: rejected, screen, "
                    "interview, no_response, offer. Null searches everything.")


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    attempts: int


class IngestRequest(BaseModel):
    company: str
    role_title: str
    raw_text: str = Field(min_length=50)
    outcome: str = "applied"
    source_url: str | None = None


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    """Serve the demo page from the app rather than as a static asset.

    Vercel routes every path to this function once vercel.json declares one, so
    a file sitting at the repo root never gets served and `/` came back as
    FastAPI's own JSON 404. Reading it here removes the question entirely, and
    the page stays an ordinary editable file rather than a Python string.
    """
    path = os.path.join(_ROOT, "index.html")
    try:
        with open(path, encoding="utf-8") as fh:
            return HTMLResponse(fh.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>RoleFit</h1><p>API is up. See "
                            "<a href='/api/health'>/api/health</a>.</p>")


@app.get("/api/health")
def health() -> dict:
    try:
        with _conn().cursor() as cur:
            cur.execute("select count(*) from jd_documents")
            docs = cur.fetchone()[0]
            cur.execute("select count(*) from jd_chunks")
            chunks = cur.fetchone()[0]
        return {"status": "ok", "documents": docs, "chunks": chunks}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


@app.post("/api/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest) -> AskResponse:
    try:
        return AskResponse(**ask(_graph(), req.question, req.outcome))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/documents")
def ingest_endpoint(req: IngestRequest) -> dict:
    try:
        n = ingest_one(_conn(), _oai(), company=req.company,
                       role=req.role_title, text=req.raw_text,
                       outcome=req.outcome, url=req.source_url)
        return {"company": req.company, "role_title": req.role_title,
                "chunks": n}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/corpus")
def corpus() -> dict:
    """What is actually in the index. Drives the counter on the demo page."""
    try:
        with _conn().cursor() as cur:
            cur.execute("""
                select outcome, count(*) from jd_documents
                group by outcome order by count(*) desc
            """)
            by_outcome = {r[0]: r[1] for r in cur.fetchall()}
            cur.execute("select company, role_title, outcome from jd_documents "
                        "order by created_at desc limit 50")
            docs = [{"company": r[0], "role": r[1], "outcome": r[2]}
                    for r in cur.fetchall()]
        return {"by_outcome": by_outcome, "documents": docs,
                "total": sum(by_outcome.values())}
    except Exception as exc:
        return {"by_outcome": {}, "documents": [], "total": 0,
                "error": str(exc)}
