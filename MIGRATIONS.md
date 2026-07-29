# Database

Schema lives in Supabase migrations, applied to project `usfxjkroohbttyntpymd`:

- `rolefit_jd_corpus` — jd_documents, jd_chunks
- `rolefit_rate_limit` — rolefit_rate_limit table + rolefit_bump_rate()
- `rolefit_switch_to_gte_small_384` — recreated chunks at vector(384) for gte-small
- `rolefit_search_rpc` — rolefit_search(), rolefit_corpus(), grants to anon

All tables have RLS enabled with no policies. Access is exclusively through the
security-definer functions above, which is a far smaller surface to reason about
than a set of row policies.
