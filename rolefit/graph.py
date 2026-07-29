"""The LangGraph agent: retrieve, grade what came back, retry or admit the gap.

This is corrective RAG rather than the straight-line version. The straight line
is retrieve then generate, which fails silently: if retrieval misses, the model
still writes a confident paragraph out of whatever it happened to get, or out of
training data. Here a grader reads the retrieved chunks first and decides whether
they can actually answer the question. If they cannot, the query is rewritten and
retrieval runs again. If it still cannot, the answer says so.

    retrieve -> grade -+-- relevant ------> generate -> END
                       |
                       +-- not relevant --> rewrite -> retrieve  (max 2)
                       |
                       +-- out of attempts -> admit_gap -> END

The admit_gap path is the point. A system that says "the corpus does not cover
this" is worth more than one that always produces prose.

Generation runs on Groq through the OpenAI-compatible endpoint, so the standard
langchain-openai client works with nothing but a different base URL.
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from . import config as cfg
from .retrieve import Chunk, format_context, search


class Grade(BaseModel):
    """Structured grader output, so the routing decision is not a vibe.

    can_answer is a yes/no string rather than a bool on purpose. Llama emits
    "false" as a string for boolean fields, and Groq's tool-call validator
    rejects the request before pydantic ever gets a chance to coerce it. A
    string enum is the shape open models reliably produce.
    """
    can_answer: Literal["yes", "no"] = Field(
        description="yes if the excerpts carry evidence bearing on the question")
    reason: str = Field(description="One sentence on what is missing, if anything.")

    @property
    def ok(self) -> bool:
        return self.can_answer == "yes"


class State(TypedDict, total=False):
    question: str
    query: str
    outcome_filter: str | None
    chunks: Annotated[list[Chunk], lambda _a, b: b]
    attempts: int
    answer: str
    grade: Grade | None


def _llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=cfg.CHAT_MODEL,
        temperature=temperature,
        api_key=cfg.require("GROQ_API_KEY", cfg.GROQ_API_KEY),
        base_url=cfg.GROQ_BASE_URL,
        timeout=45,
        max_retries=1,
    )


GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You decide whether retrieved job-description excerpts contain evidence "
     "that bears on a question.\n\n"
     "Say yes when the excerpts carry relevant evidence, even partial. Many "
     "questions here are aggregates, like which requirements appear most "
     "often. For those, excerpts listing requirements ARE the evidence; you do "
     "not need every posting in the corpus to answer, and a partial answer "
     "drawn from what was retrieved is the correct outcome.\n\n"
     "Say no only when the excerpts are genuinely off-topic, or when the "
     "question asks for a fact the excerpts simply do not contain, such as "
     "salary in a country nobody mentions. Answering from your own general "
     "knowledge rather than the excerpts is a no."),
    ("human", "Question:\n{question}\n\nExcerpts:\n{context}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions about a corpus of job descriptions one person has "
     "applied to. Every excerpt is labelled with the company, the role, and "
     "what happened to that application.\n\n"
     "Rules:\n"
     "- Use only the excerpts. If they do not support a claim, do not make it.\n"
     "- Cite with the bracket numbers, like [2].\n"
     "- When counting how often a requirement appears, say how many distinct "
     "roles you saw it in, and name them.\n"
     "- Be blunt about gaps. The person reading this wants to know where they "
     "fall short, not to be encouraged."),
    ("human", "Question:\n{question}\n\nExcerpts:\n{context}"),
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the search query so it retrieves better against job descriptions. "
     "Use the vocabulary a posting would actually use, not the vocabulary of "
     "the question. Return only the rewritten query, nothing else."),
    ("human", "Original question: {question}\nTried: {query}\nMissing: {reason}"),
])


def build_graph():
    def retrieve(state: State) -> dict:
        chunks = search(state["query"], outcome=state.get("outcome_filter"))
        return {"chunks": chunks, "attempts": state.get("attempts", 0) + 1}

    # Node names and state keys share a namespace in LangGraph, so this node
    # cannot be called "grade" while the state carries a "grade" field.
    def grade_docs(state: State) -> dict:
        if not state["chunks"]:
            return {"grade": Grade(can_answer="no",
                                   reason="retrieval returned nothing")}
        grader = _llm().with_structured_output(Grade)
        return {"grade": grader.invoke(GRADE_PROMPT.format_messages(
            question=state["question"],
            context=format_context(state["chunks"])))}

    def route(state: State) -> Literal["generate", "rewrite", "admit_gap"]:
        g = state.get("grade")
        if g is not None and g.ok:
            return "generate"
        if state.get("attempts", 0) >= cfg.MAX_RETRIEVAL_ATTEMPTS:
            return "admit_gap"
        return "rewrite"

    def rewrite(state: State) -> dict:
        g = state.get("grade")
        chain = REWRITE_PROMPT | _llm(0.3) | StrOutputParser()
        return {"query": chain.invoke({
            "question": state["question"], "query": state["query"],
            "reason": g.reason if g else "nothing retrieved"}).strip()}

    def generate(state: State) -> dict:
        chain = ANSWER_PROMPT | _llm() | StrOutputParser()
        return {"answer": chain.invoke({
            "question": state["question"],
            "context": format_context(state["chunks"])})}

    def admit_gap(state: State) -> dict:
        g = state.get("grade")
        why = g.reason if g else "nothing relevant was retrieved"
        return {"answer": ("The corpus does not cover this. " + why +
                           "\n\nEither the job descriptions in the index do not "
                           "discuss it, or there are not enough of them yet.")}

    sg = StateGraph(State)
    for name, fn in (("retrieve", retrieve), ("grade_docs", grade_docs),
                     ("rewrite", rewrite), ("generate", generate),
                     ("admit_gap", admit_gap)):
        sg.add_node(name, fn)

    sg.set_entry_point("retrieve")
    sg.add_edge("retrieve", "grade_docs")
    sg.add_conditional_edges("grade_docs", route, {
        "generate": "generate", "rewrite": "rewrite", "admit_gap": "admit_gap"})
    sg.add_edge("rewrite", "retrieve")
    sg.add_edge("generate", END)
    sg.add_edge("admit_gap", END)
    return sg.compile()


def ask(app, question: str, outcome: str | None = None) -> dict:
    final = app.invoke({"question": question, "query": question,
                        "outcome_filter": outcome, "chunks": [],
                        "attempts": 0, "answer": ""})
    return {"answer": final["answer"],
            "sources": [c.cite() for c in final["chunks"]],
            # The excerpt text, not just the labels. An eval judge that only
            # sees "Company / Role" cannot verify a single claim, which makes
            # any faithfulness score computed from labels meaningless.
            "context": format_context(final["chunks"]),
            "attempts": final["attempts"]}
