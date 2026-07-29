"""Gap analysis: what the corpus demands that a resume does not evidence.

This is the question the whole project exists to answer, minus the outcome data
it does not have yet. Rather than comparing rejections against replies, it
compares what the market asks for against what one resume actually shows.

Two passes, deliberately:

  1. EXTRACT   Pull a flat list of concrete requirements out of each job
               description, one document at a time. Doing this per document
               rather than over the whole corpus keeps the model from silently
               merging two companies' requirements into one invented phrase, and
               it means every requirement keeps a traceable source.

  2. MATCH     Check each distinct requirement against the resume text. This is
               deterministic-first: an exact or near-exact string hit counts as
               covered without asking a model anything, because a model asked
               "does this resume show Python" will say yes to almost anything.
               Only genuinely ambiguous cases go to the LLM.

The ranking is by how many distinct roles ask for a thing. A requirement that
appears in six postings and nowhere in the resume is worth more attention than
one that appears once, and that ordering falls out of the data rather than
anybody's opinion.
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from . import config as cfg
from . import supabase as sb


class Requirements(BaseModel):
    """Requirements lifted from a single posting."""
    items: list[str] = Field(
        description="Concrete requirements, each a short noun phrase such as "
                    "'Python', 'PostgreSQL', 'incident response', 'LangChain'. "
                    "Skills, tools and capabilities only. Never seniority, "
                    "location, salary, visa or degree.")


@dataclass
class Gap:
    requirement: str
    roles: list[str] = field(default_factory=list)
    covered: bool = False
    evidence: str = ""

    @property
    def demand(self) -> int:
        return len(self.roles)


EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Extract only the HARD requirements from a job posting: named "
     "technologies, tools, languages, platforms, frameworks, and specific "
     "technical practices. Short noun phrases, using the posting's own words. "
     "Split compounds, so 'PostgreSQL and/or MongoDB' becomes two items.\n\n"
     "Exclude, without exception:\n"
     "- soft skills and traits: communication, judgment, ownership, "
     "reliability, composure, attention to detail, problem solving\n"
     "- seniority, years of experience, degrees, location, salary, visa\n"
     "- vague business nouns: startups, stakeholders, fast-paced environments\n"
     "- languages spoken, such as English or Hindi\n\n"
     "A good item is something a person either has built with or has not. "
     "'LangChain' qualifies. 'Strong communication' does not."),
    ("human", "{doc}"),
])

# Belt and braces. The prompt above asks for hard requirements only and mostly
# obeys, but a single soft skill slipping through outranks a real gap in the
# report, so anything matching these is dropped after extraction. A resume
# cannot evidence "judgment", and pretending it can produces advice nobody can
# act on.
SOFT_SKILL_PAT = re.compile(
    r"\b(communicat|judgment|judgement|ownership|reliab|composure|attention|"
    r"problem[- ]solv|critical thinking|interpersonal|organi[sz]ational|"
    r"analytical|adaptab|collaborat|self[- ]start|initiative|detail|"
    r"time management|multitask|team ?(player|work)|work ethic|proactiv|"
    r"english|hindi|fluent|verbal|written|presentation skills|"
    r"stakeholder|fast[- ]paced|startups?$|ambiguity|discretion|"
    r"confidential|professional|mindset|generalist|focus)\b", re.I)


def _is_hard_requirement(req: str) -> bool:
    if SOFT_SKILL_PAT.search(req):
        return False
    if len(req) < 2 or len(req) > 45:
        return False
    # A bare adjective or a sentence fragment is not a requirement.
    return len(req.split()) <= 5

MATCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Decide whether a resume demonstrates a specific requirement.\n\n"
     "Say yes only for real evidence: the thing named, or something a "
     "practitioner would accept as the same capability under a different name. "
     "Postgres and PostgreSQL are the same. Supabase implies PostgreSQL. n8n "
     "workflow building is automation. Next.js implies React.\n\n"
     "Say no when the resume merely works in an adjacent area. Building LLM "
     "pipelines is not evidence of Kubernetes. Writing SQL is not evidence of "
     "data modelling at scale. A gap wrongly marked covered is worse than one "
     "wrongly flagged, because it is the one that fails in an interview."),
    ("human", "Requirement: {req}\n\nResume:\n{resume}"),
])


class Match(BaseModel):
    covered: str = Field(description="yes or no")
    evidence: str = Field(default="", description="The phrase in the resume that shows it, if any.")


def _llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(model=cfg.CHAT_MODEL, temperature=temperature,
                      api_key=cfg.require("GROQ_API_KEY", cfg.GROQ_API_KEY),
                      base_url=cfg.GROQ_BASE_URL, timeout=60, max_retries=1)


def _normalise(s: str) -> str:
    return re.sub(r"[^a-z0-9+#. ]", " ", s.lower()).strip()


def _direct_hit(req: str, resume_low: str) -> str:
    """Exact or near-exact presence, no model involved.

    Cheap, deterministic, and immune to a model's eagerness to agree. Only what
    this cannot settle is worth an LLM call.
    """
    r = _normalise(req)
    if not r:
        return ""
    if r in resume_low:
        return req
    # Multi-word requirements where every token is present somewhere still count;
    # "row level security" should match "row-level security".
    toks = [t for t in r.split() if len(t) > 2]
    if toks and all(t in resume_low for t in toks):
        return req
    return ""


def extract_requirements(docs: list[dict]) -> dict[str, list[str]]:
    """{requirement: [roles that ask for it]} across the corpus."""
    chain = EXTRACT_PROMPT | _llm().with_structured_output(Requirements)
    by_req: dict[str, list[str]] = defaultdict(list)
    for d in docs:
        label = f"{d['company']} / {d['role_title']}"
        # Never swallow this. A failed extraction silently produces a report
        # claiming a posting asks for nothing, which reads as "no gaps here"
        # rather than "this did not run". That is exactly how a rate limit
        # turned into a confidently empty gap report once already.
        try:
            got = chain.invoke({"doc": d["raw_text"][:6000]})
        except Exception as exc:
            raise RuntimeError(
                f"Requirement extraction failed on {label}: "
                f"{type(exc).__name__}: {str(exc)[:200]}\n"
                "The report would be wrong rather than incomplete, so it stops "
                "here."
            ) from exc
        seen = set()
        for item in got.items:
            key = item.strip().rstrip(".").lower()
            if not key or key in seen or not _is_hard_requirement(key):
                continue
            seen.add(key)
            by_req[key].append(label)
    return by_req


def analyse(resume_text: str, docs: list[dict], min_demand: int = 1,
            roles_like: str | None = None) -> list[Gap]:
    """Compare a resume against the corpus.

    roles_like narrows the corpus to postings whose title matches, because
    aggregating across unrelated roles produces noise rather than signal. A
    corpus holding both sales and AI engineering postings will report
    'cold calling' as a gap on an engineer's resume, which is true and useless.
    """
    if roles_like:
        pat = re.compile(roles_like, re.I)
        docs = [d for d in docs if pat.search(d["role_title"])
                or pat.search(d["company"])]
    by_req = extract_requirements(docs)
    resume_low = _normalise(resume_text)

    gaps: list[Gap] = []
    ambiguous: list[Gap] = []
    for req, roles in by_req.items():
        if len(roles) < min_demand:
            continue
        g = Gap(requirement=req, roles=sorted(set(roles)))
        hit = _direct_hit(req, resume_low)
        if hit:
            g.covered, g.evidence = True, "direct match"
            gaps.append(g)
        else:
            ambiguous.append(g)

    # Only what the string check could not settle costs a model call.
    matcher = MATCH_PROMPT | _llm().with_structured_output(Match)
    for g in ambiguous:
        try:
            m = matcher.invoke({"req": g.requirement, "resume": resume_text[:6000]})
            g.covered = m.covered.strip().lower() == "yes"
            g.evidence = m.evidence
        except Exception as exc:
            # Flagged, not hidden. An unchecked requirement reported as a gap is
            # recoverable; one reported as covered is not.
            g.covered = False
            g.evidence = f"UNCHECKED ({type(exc).__name__})"
        gaps.append(g)

    gaps.sort(key=lambda g: (-g.demand, g.covered, g.requirement))
    return gaps


def corpus_documents() -> list[dict]:
    return sb.rpc("rolefit_documents_full", {})


def to_json(gaps: list[Gap]) -> str:
    return json.dumps([{"requirement": g.requirement, "demand": g.demand,
                        "covered": g.covered, "roles": g.roles,
                        "evidence": g.evidence} for g in gaps], indent=2)
