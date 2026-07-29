"""What the corpus asks for that a resume does not show.

    python scripts/gap_report.py                          # against the AI Engineer resume
    python scripts/gap_report.py --resume path\\to.txt     # any plain-text resume
    python scripts/gap_report.py --min-demand 2           # only things 2+ roles want
    python scripts/gap_report.py --json                   # machine-readable

Reads the plain-text resume the resume-system build emits, so the two projects
stay in sync: rebuild the resume, rerun this, and the gaps reflect the resume
you would actually send.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rolefit import gaps as G  # noqa: E402

DEFAULT_RESUME = (r"C:\Users\Admin\Downloads\Pari_Resumes"
                  r"\Pari_Goyal_AI_Engineer.txt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default=DEFAULT_RESUME)
    ap.add_argument("--min-demand", type=int, default=1)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--roles", default=None,
                    help="regex narrowing the corpus by role title, "
                         "e.g. \"engineer|developer\". Aggregating across "
                         "unrelated roles produces noise.")
    args = ap.parse_args()

    if not os.path.exists(args.resume):
        print(f"Resume not found: {args.resume}\n"
              "Build it first: cd resume-system && python build.py")
        return 1

    with open(args.resume, encoding="utf-8") as fh:
        resume = fh.read()

    docs = G.corpus_documents()
    if not docs:
        print("Corpus is empty. Ingest some job descriptions first.")
        return 1

    print(f"Analysing {len(docs)} postings against "
          f"{os.path.basename(args.resume)}...\n")
    result = G.analyse(resume, docs, min_demand=args.min_demand,
                       roles_like=args.roles)

    if args.json:
        print(G.to_json(result))
        return 0

    missing = [g for g in result if not g.covered]
    covered = [g for g in result if g.covered]

    print("NOT EVIDENCED IN THE RESUME")
    print("-" * 74)
    if not missing:
        print("  nothing")
    for g in missing:
        roles = ", ".join(r.split(" / ")[0] for r in g.roles[:3])
        more = f" +{len(g.roles) - 3}" if len(g.roles) > 3 else ""
        print(f"  {g.demand:>2}x  {g.requirement[:42]:44s} {roles}{more}")

    print()
    print(f"COVERED  ({len(covered)} of {len(result)})")
    print("-" * 74)
    for g in covered[:12]:
        print(f"  {g.demand:>2}x  {g.requirement[:42]:44s} {g.evidence[:24]}")
    if len(covered) > 12:
        print(f"  ... and {len(covered) - 12} more")

    print()
    top = [g for g in missing if g.demand >= 2]
    if top:
        print("Worth acting on, wanted by 2+ roles and absent from the resume:")
        for g in top:
            print(f"  - {g.requirement}")
    else:
        print("Nothing is wanted by 2+ roles and missing. Either the corpus is "
              "too small\nto show a pattern, or the resume already covers the "
              "common ground.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
