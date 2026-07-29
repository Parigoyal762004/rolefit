"""Ingest a job description: chunk it, embed it, store it.

Local only. The deployed app is read-only by design, so this needs the Supabase
secret key, which should live in your local .env and nowhere else.

    python -m rolefit.ingest --file data/anthropic__ai-engineer.txt \\
        --company Anthropic --role "AI Engineer" --outcome rejected

    python -m rolefit.ingest --dir data/ --outcome rejected

Files in --dir mode are named `company__role-title.txt`.
"""

import argparse
import os
import re

from . import config as cfg
from . import supabase as sb

EMBED_BATCH = 64  # the Edge Function caps a single request at 128


def chunk(text: str, size: int = cfg.CHUNK_SIZE,
          overlap: int = cfg.CHUNK_OVERLAP) -> list[str]:
    """Split on paragraph boundaries, then pack up to `size`.

    Deliberately not a fixed character split. Job descriptions carry meaning in
    blocks, and a requirements list cut in half retrieves badly for both halves.
    This packs whole paragraphs until the next would overflow, so chunk
    boundaries land where the document already had them.
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
        while len(p) > size:
            chunks.append(p[:size])
            p = p[size - overlap:]
        buf = p
    if buf:
        chunks.append(buf)

    # Carry a tail of the previous chunk forward so a sentence spanning a
    # boundary stays retrievable from either side.
    if overlap <= 0 or len(chunks) < 2:
        return chunks
    return [chunks[0]] + [chunks[i - 1][-overlap:] + "\n" + chunks[i]
                          for i in range(1, len(chunks))]


def embed_all(texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        out.extend(sb.embed(texts[i:i + EMBED_BATCH]))
    return out


def ingest_one(*, company: str, role: str, text: str,
               outcome: str = "applied", url: str | None = None) -> int:
    chunks = chunk(text)
    vectors = embed_all(chunks)

    doc = sb.table_insert("jd_documents", [{
        "company": company, "role_title": role, "source_url": url,
        "outcome": outcome, "raw_text": text,
    }], on_conflict="company,role_title")
    doc_id = doc[0]["id"]

    # Re-ingest replaces chunks wholesale. Cheaper to reason about than diffing,
    # and the corpus is small enough that it does not matter.
    sb.table_delete("jd_chunks", {"document_id": doc_id})
    sb.table_insert("jd_chunks", [
        {"document_id": doc_id, "chunk_index": i, "content": c,
         "embedding": v}
        for i, (c, v) in enumerate(zip(chunks, vectors))
    ], returning="minimal")
    return len(chunks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("--dir")
    ap.add_argument("--company")
    ap.add_argument("--role")
    ap.add_argument("--url")
    ap.add_argument("--outcome", default="applied")
    args = ap.parse_args()

    cfg.require("SUPABASE_SECRET_KEY", cfg.SUPABASE_SECRET_KEY)

    if args.dir:
        for name in sorted(os.listdir(args.dir)):
            if not name.endswith(".txt"):
                continue
            company, _, role = name[:-4].partition("__")
            with open(os.path.join(args.dir, name), encoding="utf-8") as fh:
                text = fh.read()
            n = ingest_one(company=company,
                           role=role.replace("-", " ").title() or "Unknown",
                           text=text, outcome=args.outcome)
            print(f"{company} / {role}: {n} chunks")
    else:
        with open(args.file, encoding="utf-8") as fh:
            text = fh.read()
        n = ingest_one(company=args.company, role=args.role, text=text,
                       outcome=args.outcome, url=args.url)
        print(f"{args.company} / {args.role}: {n} chunks")


if __name__ == "__main__":
    main()
