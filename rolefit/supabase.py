"""Thin Supabase client over PostgREST and Edge Functions.

Deliberately not a Postgres connection. Serverless functions open connections
and then get frozen mid-flight, which exhausts a Postgres pool fast and produces
the worst class of bug: intermittent, load-dependent, and invisible locally.
PostgREST is stateless HTTP, so there is no connection state to get wrong, and
it means the deployed app needs only a publishable key rather than a database
password.

Everything the app reads goes through security-definer SQL functions, so RLS
stays on with no policies and the exposed surface is exactly three functions.
"""

import httpx

from . import config as cfg


class SupabaseError(RuntimeError):
    pass


def _headers(key: str) -> dict:
    return {"apikey": key, "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"}


def rpc(name: str, payload: dict, *, write: bool = False):
    """Call a Postgres function through PostgREST."""
    key = (cfg.require("SUPABASE_SECRET_KEY", cfg.SUPABASE_SECRET_KEY)
           if write else cfg.SUPABASE_PUBLISHABLE_KEY)
    url = f"{cfg.SUPABASE_URL}/rest/v1/rpc/{name}"
    r = httpx.post(url, json=payload, headers=_headers(key),
                   timeout=cfg.HTTP_TIMEOUT)
    if r.status_code >= 400:
        raise SupabaseError(f"{name}: {r.status_code} {r.text[:300]}")
    return r.json()


def table_insert(table: str, rows: list[dict], *, on_conflict: str | None = None,
                 returning: str = "representation"):
    """Insert or upsert rows. Writes require the secret key, so this is local only."""
    key = cfg.require("SUPABASE_SECRET_KEY", cfg.SUPABASE_SECRET_KEY)
    url = f"{cfg.SUPABASE_URL}/rest/v1/{table}"
    params = {}
    headers = _headers(key)
    headers["Prefer"] = f"return={returning}"
    if on_conflict:
        params["on_conflict"] = on_conflict
        headers["Prefer"] += ",resolution=merge-duplicates"
    r = httpx.post(url, json=rows, headers=headers, params=params,
                   timeout=cfg.HTTP_TIMEOUT)
    if r.status_code >= 400:
        raise SupabaseError(f"insert {table}: {r.status_code} {r.text[:300]}")
    return r.json() if returning == "representation" else None


def table_delete(table: str, filters: dict):
    key = cfg.require("SUPABASE_SECRET_KEY", cfg.SUPABASE_SECRET_KEY)
    url = f"{cfg.SUPABASE_URL}/rest/v1/{table}"
    params = {k: f"eq.{v}" for k, v in filters.items()}
    r = httpx.delete(url, headers=_headers(key), params=params,
                     timeout=cfg.HTTP_TIMEOUT)
    if r.status_code >= 400:
        raise SupabaseError(f"delete {table}: {r.status_code} {r.text[:300]}")


def embed(texts: list[str]) -> list[list[float]]:
    """Embed via the Edge Function running gte-small on Supabase's infrastructure.

    Batched in the caller, not here, because the Edge Function caps a single
    request at 128 inputs.
    """
    url = f"{cfg.SUPABASE_URL}/functions/v1/embed"
    r = httpx.post(url, json={"input": texts},
                   headers=_headers(cfg.SUPABASE_PUBLISHABLE_KEY),
                   timeout=cfg.HTTP_TIMEOUT)
    if r.status_code >= 400:
        raise SupabaseError(f"embed: {r.status_code} {r.text[:300]}")
    data = r.json()
    if "embeddings" not in data:
        raise SupabaseError(f"embed returned no embeddings: {str(data)[:200]}")
    return data["embeddings"]


def embed_one(text: str) -> list[float]:
    return embed([text])[0]
