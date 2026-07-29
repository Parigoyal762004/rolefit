"""Pull outcomes from the live applications table into the JD corpus.

The whole premise of RoleFit is comparing what the rejections asked for against
what the replies asked for. That comparison needs outcomes, and outcomes already
live in the `applications` table that the job-outreach tool writes to. Rather
than maintaining a second copy by hand, this syncs one into the other.

    python scripts/sync_outcomes.py            # show what would change
    python scripts/sync_outcomes.py --apply    # write it

Needs ROLEFIT_SUPABASE_SECRET_KEY in .env, because it writes.

Update `status` in the outreach tool when you hear back, run this, and the
comparative questions start working. Until then RoleFit can only answer what
the market asks for, not why any particular application failed.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx  # noqa: E402

from rolefit import config as cfg  # noqa: E402
from rolefit import supabase as sb  # noqa: E402

# applications.status uses the outreach tool's vocabulary. jd_documents.outcome
# uses the corpus vocabulary. They overlap but are not identical, so the mapping
# is explicit rather than assumed.
STATUS_MAP = {
    "to_apply": "to_apply",
    "applied": "applied",
    "no_response": "no_response",
    "rejected": "rejected",
    "screen": "screen",
    "interview": "interview",
    "offer": "offer",
    # 'skipped' means the application was never sent, so there is nothing to
    # learn from it. Deliberately not mapped; those rows are left alone.
}


def fetch(table: str, select: str) -> list[dict]:
    r = httpx.get(f"{cfg.SUPABASE_URL}/rest/v1/{table}",
                  params={"select": select},
                  headers={"apikey": cfg.SUPABASE_PUBLISHABLE_KEY,
                           "Authorization": f"Bearer {cfg.SUPABASE_PUBLISHABLE_KEY}"},
                  timeout=cfg.HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


def norm(s: str) -> str:
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write the changes; otherwise this is a dry run")
    args = ap.parse_args()

    try:
        apps = fetch("applications", "company,role,status")
    except httpx.HTTPStatusError:
        print("Cannot read the applications table with the publishable key.\n"
              "It has RLS enabled and no policy for anon, which is correct. Run\n"
              "this with ROLEFIT_SUPABASE_SECRET_KEY set instead.")
        return 1

    docs = sb.rpc("rolefit_corpus", {})

    by_company: dict[str, str] = {}
    for a in apps:
        mapped = STATUS_MAP.get((a.get("status") or "").strip())
        if mapped:
            by_company[norm(a.get("company"))] = mapped

    changes = []
    for d in docs:
        want = by_company.get(norm(d["company"]))
        if want and want != d["outcome"]:
            changes.append((d["company"], d["role_title"], d["outcome"], want))

    if not changes:
        print(f"{len(docs)} documents, {len(by_company)} applications matched. "
              "Nothing to change.")
        counts: dict[str, int] = {}
        for d in docs:
            counts[d["outcome"]] = counts.get(d["outcome"], 0) + 1
        print("current outcomes:", ", ".join(f"{k}={v}" for k, v in counts.items()))
        if set(counts) <= {"to_apply", "applied"}:
            print("\nNo document has an outcome past 'applied', so the "
                  "comparative questions\ncannot be answered yet. That is a data "
                  "gap, not a bug.")
        return 0

    print(f"{len(changes)} document(s) would change:\n")
    for company, role, old, new in changes:
        print(f"  {company:16s} {role:38s} {old:12s} -> {new}")

    if not args.apply:
        print("\nDry run. Re-run with --apply to write.")
        return 0

    for company, role, _old, new in changes:
        sb.table_insert("jd_documents", [{"company": company,
                                          "role_title": role,
                                          "outcome": new}],
                        on_conflict="company,role_title", returning="minimal")
    print(f"\nUpdated {len(changes)} document(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
