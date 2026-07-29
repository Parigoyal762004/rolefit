"""One-off seeding of the JD corpus from data/.

Differs from `python -m rolefit.ingest --dir` only in that the outcome is per
file rather than one value for the whole directory, because these came out of
the live applications table with different statuses each.

    python scripts/seed.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rolefit.ingest import ingest_one  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data")

# Real status from the applications table, not invented. Nothing here claims an
# outcome that did not happen: 'to_apply' means exactly that.
OUTCOMES = {
    "Peakflo": "applied",
    "Vela": "applied",
    "Browser-Use": "to_apply",
    "Shadeform": "to_apply",
    "Weekday": "to_apply",
    "Clinikally": "to_apply",
    "Enerjazz": "to_apply",
    "SarvM-AI": "to_apply",
    "ZenTrades": "to_apply",
}


def main() -> None:
    total = 0
    for name in sorted(os.listdir(DATA)):
        if not name.endswith(".txt"):
            continue
        company, _, role_slug = name[:-4].partition("__")
        role = role_slug.replace("-", " ").title()
        with open(os.path.join(DATA, name), encoding="utf-8") as fh:
            text = fh.read()
        n = ingest_one(company=company.replace("-", " "), role=role, text=text,
                       outcome=OUTCOMES.get(company, "applied"))
        total += n
        print(f"{company:14s} {role:42s} {n:2d} chunks")
    print(f"\n{total} chunks total")


if __name__ == "__main__":
    main()
