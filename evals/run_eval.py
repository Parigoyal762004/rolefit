"""Run the eval set and score it.

    python -m evals.run_eval

Three metrics, and the third is the one that matters:

  citation_ok    Did the answer cite at least the expected number of sources?
                 Cheap proxy for "did retrieval actually find anything".

  faithfulness   LLM-as-judge: is every claim in the answer supported by the
                 retrieved excerpts? This is the number that catches the failure
                 mode where the model answers correctly from world knowledge
                 while retrieval quietly returned nothing useful.

  gap_honesty    On questions the corpus genuinely cannot answer, did the system
                 say so instead of inventing an answer? A RAG system that scores
                 well here is worth more than one that scores well on the rest,
                 because in production the out-of-corpus question is the common
                 case, not the edge case.

Run this before and after any retrieval change. Without it you are tuning
chunk sizes on a feeling.
"""

import json
import os
import re
import sys

from langchain_anthropic import ChatAnthropic
from openai import OpenAI
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rolefit import config as cfg          # noqa: E402
from rolefit.graph import ask, build_graph  # noqa: E402
from rolefit.ingest import connect          # noqa: E402
from rolefit.retrieve import format_context  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


class Judgement(BaseModel):
    supported: bool = Field(description="Is every factual claim traceable to the excerpts?")
    unsupported_claim: str = Field(default="", description="The first claim that is not.")


JUDGE_SYSTEM = (
    "You check whether an answer is grounded in the excerpts it was given. "
    "You are not judging whether the answer is correct in the world. You are "
    "judging whether every claim in it can be traced to the excerpts. An answer "
    "that is true but not supported by the excerpts is a failure, because it "
    "means the system is answering from training data and retrieval is not "
    "doing its job."
)

GAP_PHRASES = ("does not cover", "not enough", "no information", "cannot answer",
               "not covered", "have not ingested")


def main() -> int:
    with open(os.path.join(HERE, "eval_set.json"), encoding="utf-8") as fh:
        cases = json.load(fh)

    conn = connect()
    oai = OpenAI(api_key=cfg.require("OPENAI_API_KEY", cfg.OPENAI_API_KEY))
    graph = build_graph(conn, oai)
    judge = ChatAnthropic(model=cfg.CHAT_MODEL, temperature=0,
                          api_key=cfg.ANTHROPIC_API_KEY
                          ).with_structured_output(Judgement)

    rows, faithful, cited, honest, n_gap = [], 0, 0, 0, 0

    for case in cases:
        res = ask(graph, case["question"], case.get("outcome"))
        answer, sources = res["answer"], res["sources"]
        admitted = any(p in answer.lower() for p in GAP_PHRASES)

        if case.get("expect_gap"):
            n_gap += 1
            ok = admitted
            honest += ok
            rows.append((case["id"], "gap-honesty", "pass" if ok else "FAIL",
                         "" if ok else "invented an answer"))
            continue

        n_cites = len(set(sources))
        cite_ok = n_cites >= case.get("must_cite_at_least", 1)
        cited += cite_ok

        missing = [m for m in case.get("must_mention", [])
                   if m.lower() not in answer.lower()]

        j = judge.invoke([("system", JUDGE_SYSTEM),
                          ("human", f"Excerpts:\n{chr(10).join(sources)}\n\n"
                                    f"Answer:\n{answer}")])
        faithful += j.supported

        status = "pass" if (cite_ok and j.supported and not missing) else "FAIL"
        detail = []
        if not cite_ok:
            detail.append(f"only {n_cites} sources")
        if not j.supported:
            detail.append(f"unsupported: {j.unsupported_claim[:60]}")
        if missing:
            detail.append(f"missing terms: {', '.join(missing)}")
        rows.append((case["id"], "grounded-qa", status, "; ".join(detail)))

    n_qa = len(cases) - n_gap
    print("\n%-20s %-14s %-6s %s" % ("case", "kind", "result", "detail"))
    print("-" * 78)
    for r in rows:
        print("%-20s %-14s %-6s %s" % r)
    print("-" * 78)
    if n_qa:
        print("citation coverage  %d/%d" % (cited, n_qa))
        print("faithfulness       %d/%d" % (faithful, n_qa))
    if n_gap:
        print("gap honesty        %d/%d" % (honest, n_gap))

    conn.close()
    failed = sum(1 for r in rows if r[2] == "FAIL")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
