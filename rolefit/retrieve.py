"""Retrieval. The hybrid search itself lives in SQL; this calls it and shapes the result.

Why hybrid. Dense embeddings match "owns the system end to end" against "full
ownership of the stack" where no words overlap at all. They are bad at rare exact
tokens: LangGraph, pgvector and MCP all sit near each other in embedding space
and near a hundred other tool names too. Postgres full-text is the opposite.
Running both and fusing the ranks gets the union of their strengths.

Reciprocal Rank Fusion does the fusing: each chunk scores sum(1 / (k + rank))
across the lists it appears in. It needs only each retriever's ordering, never
their scores, which is exactly why it works when one side returns cosine
distance and the other returns ts_rank.

See the rolefit_search function in the migrations for the SQL.
"""

from dataclasses import dataclass

from . import config as cfg
from . import supabase as sb


@dataclass
class Chunk:
    chunk_id: int
    document_id: int
    company: str
    role_title: str
    outcome: str
    content: str
    score: float

    def cite(self) -> str:
        return f"{self.company} / {self.role_title} [{self.outcome.replace('_', ' ')}]"


def search(question: str, *, k: int = cfg.TOP_K,
           outcome: str | None = None) -> list[Chunk]:
    """Retrieve the k best chunks.

    `outcome` filters to one application result, and it is what makes the
    comparative questions work: retrieve only from rejections, retrieve only
    from the ones that reached a screen, then ask what differs.
    """
    rows = sb.rpc("rolefit_search", {
        "q_embedding": sb.embed_one(question),
        "q_text": question,
        "q_outcome": outcome,
        "n_candidates": cfg.CANDIDATES_PER_RETRIEVER,
        "n_results": k,
        "rrf_k": cfg.RRF_K,
    })
    return [Chunk(r["chunk_id"], r["document_id"], r["company"],
                  r["role_title"], r["outcome"], r["content"], r["score"])
            for r in rows]


def format_context(chunks: list[Chunk]) -> str:
    """Number the chunks so the model can cite them and the citation is checkable."""
    return "\n\n".join(
        f"[{i + 1}] {c.cite()}\n{c.content}" for i, c in enumerate(chunks)
    )
