# RoleFit

Retrieval over the job descriptions you actually applied to, so you can ask what
the rejections have in common.

I was applying to a lot of AI engineering roles and getting a lot of skill
mismatch rejections without ever being told which skill. The job descriptions
were sitting in my browser history the whole time, so I put them in a database
and asked them directly.

```
"What do the roles that rejected me ask for that the others do not?"

Retrieval-augmented generation appears in 7 of the 9 rejections and 1 of the 4
that replied [1][3][6]. Vector databases, usually Pinecone or pgvector, appear in
6 [1][2][6]. Both are absent from every role that reached a screen, which all
lead on shipping and integration instead [4][5].
```

## Why it is built the way it is

**Hybrid retrieval.** Dense embeddings match "owns the system end to end" against
"full ownership of the stack" where no words overlap. They are bad at rare exact
tokens, because `LangGraph`, `pgvector` and `MCP` all sit near each other and near
a hundred other tool names. Postgres full-text is the opposite. Both run, and the
rankings are fused with Reciprocal Rank Fusion, which only needs each retriever's
ordering and so sidesteps having to weight cosine distance against `ts_rank`.

**Corrective, not straight-line.** Retrieve then generate fails silently: if
retrieval misses, the model still writes a confident paragraph from training
data. Here a grader reads the retrieved chunks first and decides whether they can
answer the question. If not, the query is rewritten and retrieval runs again. If
it still cannot, the system says the corpus does not cover it.

```
retrieve -> grade -+-- relevant ------> generate -> END
                   +-- not relevant --> rewrite -> retrieve   (max 2)
                   +-- out of attempts -> admit_gap -> END
```

**Paragraph-aware chunking.** Job descriptions carry meaning in blocks: company
blurb, then requirements, then benefits. A fixed character split cuts a
requirements list in half and both halves retrieve badly. This packs whole
paragraphs up to the window, so boundaries land where the document already had
them.

**The outcome column.** Every document is tagged with what happened to that
application. Without it this is a search box. With it, retrieval can be filtered
to just the rejections, which is the only reason the comparative questions work.

## Stack

FastAPI, LangChain and LangGraph, Postgres with pgvector on Supabase, OpenAI
`text-embedding-3-small` for embeddings, Claude for generation and grading.

## Running it

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env          # then fill in the three values
```

Paste `schema.sql` into the Supabase SQL editor and run it. Then:

```bash
# one JD
python -m rolefit.ingest --file data/anthropic__ai-engineer.txt \
    --company Anthropic --role "AI Engineer" --outcome rejected

# or a folder of company__role.txt files
python -m rolefit.ingest --dir data/

uvicorn rolefit.api:app --reload
```

```bash
curl -X POST localhost:8000/ask -H "content-type: application/json" \
  -d "{\"question\":\"which requirements show up most often?\"}"
```

## Evals

```bash
python -m evals.run_eval
```

Three metrics. Citation coverage is a cheap proxy for whether retrieval found
anything. Faithfulness is LLM-as-judge on whether every claim traces to a
retrieved excerpt, which catches the failure where the model answers correctly
from world knowledge while retrieval returned nothing. Gap honesty is the one
that matters: on questions the corpus cannot answer, did the system say so, or
did it invent something.

The `out-of-corpus` row in `eval_set.json` is the most important test in the
file. Run this before and after any retrieval change, otherwise you are tuning
chunk sizes on a feeling.

## Where this goes next

The corpus already knows what each company asks for. The obvious next step is
generation rather than analysis: draft outreach grounded in retrieved specifics
about that company's posting, instead of a template with the name swapped.
