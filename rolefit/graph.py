"""The LangGraph agent: retrieve, grade what came back, retry or answer.

This is corrective RAG rather than the straight-line version. The straight line
is retrieve then generate, which fails silently: if retrieval misses, the model
still writes a confident paragraph out of whatever it got. Here a grader looks at
the retrieved chunks first and decides whether they can actually answer the
question. If they cannot, the query gets rewritten and retrieval runs again. If
it still cannot, the answer says so rather than inventing one.

    retrieve -> grade -+-- relevant ------> generate -> END
                       |
                       +-- not relevant --> rewrite -> retrieve  (max 2 times)
                       |
                       +-- out of attempts -> admit_gap -> END

The admit_gap path is the one that matters. A system that says "the corpus does
not cover this" is more useful than one that always produces prose.
"""

from typing import Annotated, Literal, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from . import config as cfg
from .retrieve import Chunk, format_context, search


class State(TypedDict, total=False):
    question: str
    query: str                      # the possibly-rewritten search query
    outcome_filter: str | None
    chunks: Annotated[list[Chunk], lambda a, b: b]
    attempts: int
    answer: str
    grade: "Grade | None"


class Grade(BaseModel):
    """Structured output from the grader, so the routing decision is not a vibe."""
    can_answer: bool = Field(description="Do these chunks contain enough to answer?")
    reason: str = Field(description="One sentence on what is missing, if anything.")


def _llm(temperature: float = 0.0):
    return ChatAnthropic(model=cfg.CHAT_MODEL, temperature=temperature,
                         api_key=cfg.ANTHROPIC_API_KEY)


GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You grade retrieved job-description excerpts against a question. "
     "Be strict. Excerpts that are merely on-topic do not count; they have to "
     "contain the specific information the question asks for. Answering from "
     "general knowledge is a failure, not a success."),
    ("human", "Question:\n{question}\n\nExcerpts:\n{context}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions about a corpus of job descriptions someone has "
     "applied to. Every excerpt is labelled with the company, the role, and "
     "what happened to that application.\n\n"
     "Rules:\n"
     "- Use only the excerpts. If they do not support a claim, do not make it.\n"
     "- Cite with the bracket numbers, like [2].\n"
     "- When you are counting how often a requirement appears, say how many "
     "distinct roles you saw it in, and name them.\n"
     "- Be blunt about gaps. The person reading this wants to know where they "
     "fall short, not to feel encouraged."),
    ("human", "Question:\n{question}\n\nExcerpts:\n{context}"),
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the search query to retrieve better against job descriptions. "
     "Use the vocabulary a job posting would actually use, not the vocabulary "
     "of the question. Return only the rewritten query."),
    ("human", "Original question: {question}\nTried: {query}\nMissing: {reason}"),
])


def build_graph(conn, oai):
    def retrieve(state: State) -> dict:
        chunks = search(conn, oai, state["query"],
                        outcome=state.get("outcome_filter"))
        return {"chunks": chunks, "attempts": state.get("attempts", 0) + 1}

    def grade(state: State) -> dict:
        if not state["chunks"]:
            return {"grade": Grade(can_answer=False,
                                   reason="retrieval returned nothing")}
        grader = _llm().with_structured_output(Grade)
        g = grader.invoke(GRADE_PROMPT.format_messages(
            question=state["question"],
            context=format_context(state["chunks"])))
        return {"grade": g}

    def route(state: State) -> Literal["generate", "rewrite", "admit_gap"]:
        g = state.get("grade")
        if g is not None and g.can_answer:
            return "generate"
        if state.get("attempts", 0) >= cfg.MAX_RETRIEVAL_ATTEMPTS:
            return "admit_gap"
        return "rewrite"

    def rewrite(state: State) -> dict:
        g = state.get("grade")
        chain = REWRITE_PROMPT | _llm(0.3) | StrOutputParser()
        new_q = chain.invoke({"question": state["question"],
                              "query": state["query"],
                              "reason": g.reason if g else "nothing retrieved"})
        return {"query": new_q.strip()}

    def generate(state: State) -> dict:
        chain = ANSWER_PROMPT | _llm() | StrOutputParser()
        return {"answer": chain.invoke({
            "question": state["question"],
            "context": format_context(state["chunks"])})}

    def admit_gap(state: State) -> dict:
        g = state.get("grade")
        why = g.reason if g else "nothing relevant was retrieved"
        return {"answer": (
            "The corpus does not cover this. " + why +
            "\n\nEither the job descriptions you have ingested do not discuss "
            "it, or you have not ingested enough of them yet.")}

    sg = StateGraph(State)
    sg.add_node("retrieve", retrieve)
    sg.add_node("grade", grade)
    sg.add_node("rewrite", rewrite)
    sg.add_node("generate", generate)
    sg.add_node("admit_gap", admit_gap)

    sg.set_entry_point("retrieve")
    sg.add_edge("retrieve", "grade")
    sg.add_conditional_edges("grade", route, {
        "generate": "generate", "rewrite": "rewrite", "admit_gap": "admit_gap",
    })
    sg.add_edge("rewrite", "retrieve")
    sg.add_edge("generate", END)
    sg.add_edge("admit_gap", END)
    return sg.compile()


def ask(app, question: str, outcome: str | None = None) -> dict:
    final = app.invoke({"question": question, "query": question,
                        "outcome_filter": outcome, "chunks": [], "attempts": 0,
                        "answer": ""})
    return {
        "answer": final["answer"],
        "sources": [c.cite() for c in final["chunks"]],
        "attempts": final["attempts"],
    }
