"""Hybrid retrieval: dense vectors and Postgres full-text, fused with RRF.

Why both. Dense embeddings are good at "they want someone who can own a system
end to end" matching "full ownership of the stack", where no words overlap.
They are bad at rare exact tokens: "LangGraph", "pgvector", "MCP" all sit near
each other in embedding space and near a hundred other tool names too. Full-text
search is the opposite. Running both and fusing the ranks gets you the union of
their strengths without having to tune a weight between two incomparable scores.

Reciprocal Rank Fusion is the fusing method: each document scores
sum(1 / (k + rank)) across the lists it appears in. It only needs the ordering
from each retriever, not the scores, which is exactly why it works when one
retriever returns cosine distance and the other returns ts_rank.
"""

from dataclasses import dataclass

from openai import OpenAI

from . import config as cfg


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
        return f"{self.company} / {self.role_title} [{self.outcome}]"


HYBRID_SQL = """
with dense as (
    select c.id,
           row_number() over (order by c.embedding <=> %(qvec)s) as rank
    from jd_chunks c
    join jd_documents d on d.id = c.document_id
    where (%(outcome)s is null or d.outcome = %(outcome)s)
    order by c.embedding <=> %(qvec)s
    limit %(n)s
),
sparse as (
    select c.id,
           row_number() over (
               order by ts_rank(c.tsv, websearch_to_tsquery('english', %(q)s)) desc
           ) as rank
    from jd_chunks c
    join jd_documents d on d.id = c.document_id
    where c.tsv @@ websearch_to_tsquery('english', %(q)s)
      and (%(outcome)s is null or d.outcome = %(outcome)s)
    limit %(n)s
),
fused as (
    select id, sum(score) as score from (
        select id, 1.0 / (%(rrf_k)s + rank) as score from dense
        union all
        select id, 1.0 / (%(rrf_k)s + rank) as score from sparse
    ) s
    group by id
)
select c.id, c.document_id, d.company, d.role_title, d.outcome,
       c.content, f.score
from fused f
join jd_chunks c on c.id = f.id
join jd_documents d on d.id = c.document_id
order by f.score desc
limit %(k)s
"""


def embed_query(client: OpenAI, text: str) -> list[float]:
    return client.embeddings.create(
        model=cfg.EMBED_MODEL, input=[text]
    ).data[0].embedding


def search(conn, client: OpenAI, question: str, *, k: int = cfg.TOP_K,
           outcome: str | None = None) -> list[Chunk]:
    """Retrieve the k best chunks. `outcome` filters to one application result.

    Filtering by outcome is what makes the comparative questions work: retrieve
    only from rejections, retrieve only from the ones that led to a screen, then
    ask what differs.
    """
    qvec = embed_query(client, question)
    with conn.cursor() as cur:
        cur.execute(HYBRID_SQL, {
            "qvec": qvec, "q": question, "n": cfg.CANDIDATES_PER_RETRIEVER,
            "k": k, "rrf_k": cfg.RRF_K, "outcome": outcome,
        })
        return [Chunk(*row) for row in cur.fetchall()]


def format_context(chunks: list[Chunk]) -> str:
    """Number the chunks so the model can cite them and we can check the citation."""
    return "\n\n".join(
        f"[{i + 1}] {c.cite()}\n{c.content}" for i, c in enumerate(chunks)
    )
