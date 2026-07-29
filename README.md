# RoleFit

Retrieval over the job descriptions you actually applied to, so you can ask what
the rejections have in common.

**Live: https://rolefit-wine.vercel.app**

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

**Hybrid retrieval.** Dense embeddings match "owns the system end to end"
against "full ownership of the stack" where no words overlap. They are bad at
rare exact tokens, because `LangGraph`, `pgvector` and `MCP` sit near each other
in embedding space and near a hundred other tool names too. Postgres full-text
is the opposite. Both run, and the rankings fuse with Reciprocal Rank Fusion,
which needs only each retriever's *ordering* and so sidesteps having to weight a
cosine distance against a `ts_rank`. The whole thing is one SQL function.

**Corrective, not straight-line.** Retrieve-then-generate fails silently: if
retrieval misses, the model still writes a confident paragraph from training
data. Here a grader reads the retrieved chunks first and decides whether they can
answer the question at all.

```
retrieve -> grade -+-- relevant ------> generate -> END
                   +-- not relevant --> rewrite -> retrieve   (max 2)
                   +-- out of attempts -> admit_gap -> END
```

The `admit_gap` branch is the point. Ask it something the corpus does not cover
and it says so.

**Embeddings with no embeddings vendor.** xAI has no public embeddings endpoint,
and embeddings are the load-bearing half of retrieval. Rather than add a third
API to the bill, they run on `gte-small` inside a Supabase Edge Function, on
Supabase's own infrastructure. 384 dimensions, normalised, free. It scores 61.36
on MTEB against `text-embedding-3-small`'s 62.26, which is noise at this corpus
size.

**No database connection.** Serverless functions open Postgres connections and
then get frozen mid-flight, which exhausts a pool and produces the worst class of
bug: intermittent, load-dependent, invisible locally. Everything goes through
PostgREST instead, calling security-definer SQL functions. No connection state,
and the deployed app needs only a publishable key rather than a database
password.

**Paragraph-aware chunking.** Job descriptions carry meaning in blocks: company
blurb, then requirements, then benefits. A fixed character split cuts a
requirements list in half and both halves retrieve badly. This packs whole
paragraphs up to the window so boundaries land where the document already had
them.

**The outcome column.** Every document is tagged with what happened to that
application. Without it this is a search box. With it, retrieval can be filtered
to just the rejections, which is the only reason the comparative questions work.

## Security

The deployment is public, so it is built as though people will find it.

- **Read-only.** There is no write endpoint. Ingest is a local CLI operation
  against the secret key, which never touches the deployment.
- **RLS on, zero policies.** Nothing reaches the tables through PostgREST
  directly. Three security-definer functions expose exactly what the demo needs.
- **Rate limited in Postgres, not memory.** Per-IP and global hourly ceilings.
  An in-memory counter is useless in serverless because every invocation can
  land on a fresh instance, making the real ceiling (limit x warm instances).
  The global cap is what protects the API budget: each question is two to four
  LLM calls.
- **CORS pinned** to the deployment origin and localhost. A wildcard would let
  any site on the internet spend this deployment's budget from a visitor's
  browser.
- **CSP, nosniff, DENY framing.** The page is entirely self-contained, no
  external requests, so the policy can be tight.
- **Errors are truncated** before they reach a client. Raw exceptions leak stack
  frames, table names, and sometimes fragments of keys.

## Stack

FastAPI and LangGraph on Vercel. Postgres with pgvector on Supabase. `gte-small`
in a Supabase Edge Function for embeddings. Grok 4.5 for generation and grading,
through its OpenAI-compatible endpoint.

## Running it

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

`XAI_API_KEY` is the only value the app needs to answer questions. Add
`SUPABASE_SECRET_KEY` as well if you want to ingest.

```bash
# one JD
python -m rolefit.ingest --file data/anthropic__ai-engineer.txt \
    --company Anthropic --role "AI Engineer" --outcome rejected

# or a folder of company__role.txt files
python -m rolefit.ingest --dir data/ --outcome rejected

uvicorn rolefit.api:app --reload
```

## Evals

```bash
python -m evals.run_eval
```

Citation coverage is a cheap proxy for whether retrieval found anything.
Faithfulness is LLM-as-judge on whether every claim traces to a retrieved
excerpt, which catches the failure where the model answers correctly from world
knowledge while retrieval returned nothing. Gap honesty is the one that matters:
on questions the corpus cannot answer, did the system say so.

The `out-of-corpus` row in `eval_set.json` is the most important test in the
file.

## Where this goes next

The corpus already knows what each company asks for. The obvious next step is
generation rather than analysis: draft outreach grounded in retrieved specifics
about that company's posting, instead of a template with the name swapped.
