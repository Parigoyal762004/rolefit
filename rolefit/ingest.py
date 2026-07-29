"""Ingest a job description: chunk it, embed it, store it.

Usage:
    python -m rolefit.ingest --file data/anthropic-ai-engineer.txt \\
        --company Anthropic --role "AI Engineer" --outcome rejected

Or point it at a folder of .txt files named `company__role.txt` and it will do
the lot:
    python -m rolefit.ingest --dir data/
"""

import argparse
import os
import re

import psycopg
from openai import OpenAI
from pgvector.psycopg import register_vector

from . import config as cfg


def chunk(text: str, size: int = cfg.CHUNK_SIZE,
          overlap: int = cfg.CHUNK_OVERLAP) -> list[str]:
    """Split on paragraph boundaries, then pack up to `size`.

    Deliberately not a fixed character split. Job descriptions carry their
    meaning in blocks, and a requirements list cut in half retrieves badly for
    both halves. This packs whole paragraphs until adding the next one would
    overflow, so chunk boundaries land where the document already had them.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= size:
            buf = f"{buf}\n\n{p}" if buf else p
            continue
        if buf:
            chunks.append(buf)
        # A single paragraph longer than the window still has to be broken.
        while len(p) > size:
            chunks.append(p[:size])
            p = p[size - overlap:]
        buf = p
    if buf:
        chunks.append(buf)

    # Carry a tail of the previous chunk into the next one so a sentence
    # spanning a boundary is retrievable from either side.
    if overlap <= 0 or len(chunks) < 2:
        return chunks
    return [chunks[0]] + [
        (chunks[i - 1][-overlap:] + "\n" + chunks[i]) for i in range(1, len(chunks))
    ]


def embed(client: OpenAI, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=cfg.EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def ingest_one(conn, client: OpenAI, *, company: str, role: str, text: str,
               outcome: str = "applied", url: str | None = None) -> int:
    chunks = chunk(text)
    vectors = embed(client, chunks)

    with conn.cursor() as cur:
        cur.execute(
            """
            insert into jd_documents (company, role_title, source_url,
                                      outcome, raw_text)
            values (%s, %s, %s, %s, %s)
            on conflict (company, role_title) do update
                set raw_text = excluded.raw_text,
                    outcome  = excluded.outcome,
                    source_url = excluded.source_url
            returning id
            """,
            (company, role, url, outcome, text),
        )
        doc_id = cur.fetchone()[0]

        # Re-ingest replaces chunks wholesale. Cheaper to reason about than
        # diffing, and the corpus is small.
        cur.execute("delete from jd_chunks where document_id = %s", (doc_id,))
        cur.executemany(
            """
            insert into jd_chunks (document_id, chunk_index, content, embedding)
            values (%s, %s, %s, %s)
            """,
            [(doc_id, i, c, v) for i, (c, v) in enumerate(zip(chunks, vectors))],
        )
    conn.commit()
    return len(chunks)


def connect():
    conn = psycopg.connect(cfg.require("DATABASE_URL", cfg.DATABASE_URL))
    register_vector(conn)
    return conn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("--dir")
    ap.add_argument("--company")
    ap.add_argument("--role")
    ap.add_argument("--url")
    ap.add_argument("--outcome", default="applied")
    args = ap.parse_args()

    client = OpenAI(api_key=cfg.require("OPENAI_API_KEY", cfg.OPENAI_API_KEY))
    conn = connect()

    if args.dir:
        for name in sorted(os.listdir(args.dir)):
            if not name.endswith(".txt"):
                continue
            stem = name[:-4]
            company, _, role = stem.partition("__")
            with open(os.path.join(args.dir, name), encoding="utf-8") as fh:
                text = fh.read()
            n = ingest_one(conn, client, company=company,
                           role=role.replace("-", " ") or "Unknown",
                           text=text, outcome=args.outcome)
            print(f"{company} / {role}: {n} chunks")
    else:
        with open(args.file, encoding="utf-8") as fh:
            text = fh.read()
        n = ingest_one(conn, client, company=args.company, role=args.role,
                       text=text, outcome=args.outcome, url=args.url)
        print(f"{args.company} / {args.role}: {n} chunks")

    conn.close()


if __name__ == "__main__":
    main()
