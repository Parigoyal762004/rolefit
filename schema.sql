-- RoleFit schema. Run this once in the Supabase SQL editor.
--
-- Two tables and two indexes. The interesting part is that chunks carry both an
-- embedding and a generated tsvector, because retrieval here is hybrid: dense
-- vectors find things phrased differently, sparse full-text finds exact tokens
-- like "LangGraph" or "pgvector" that embeddings blur together.

create extension if not exists vector;

create table if not exists jd_documents (
    id          bigserial primary key,
    company     text not null,
    role_title  text not null,
    source_url  text,
    applied_on  date,
    -- The outcome column is the whole point. Without it this is a document
    -- search box. With it you can ask what separates the roles that replied
    -- from the roles that did not.
    outcome     text not null default 'applied'
                check (outcome in ('applied', 'no_response', 'rejected',
                                   'screen', 'interview', 'offer')),
    raw_text    text not null,
    created_at  timestamptz not null default now(),
    unique (company, role_title)
);

create table if not exists jd_chunks (
    id           bigserial primary key,
    document_id  bigint not null references jd_documents(id) on delete cascade,
    chunk_index  int not null,
    content      text not null,
    -- text-embedding-3-small. If you switch models, change this number and
    -- re-ingest; pgvector will not let you mix dimensions in one column.
    embedding    vector(1536),
    tsv          tsvector generated always as (to_tsvector('english', content))
                 stored,
    unique (document_id, chunk_index)
);

-- HNSW over cosine distance. Faster to query than IVFFlat and does not need a
-- training step, which matters when the corpus grows one JD at a time.
create index if not exists jd_chunks_embedding_idx
    on jd_chunks using hnsw (embedding vector_cosine_ops);

create index if not exists jd_chunks_tsv_idx
    on jd_chunks using gin (tsv);

create index if not exists jd_chunks_document_idx
    on jd_chunks (document_id);
