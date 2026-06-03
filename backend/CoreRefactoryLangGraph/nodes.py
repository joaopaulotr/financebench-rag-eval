from typing import                  TypedDict, List

from langsmith import               traceable
from qdrant_client.models import    Filter, FieldCondition, MatchText

from tools import                   model, vectorstore, reranker
from prompts import                 ANALYST_SYSTEM_PROMPT, QUERY_FILTER_SYSTEM, GRADE_PROMPT

class RAGState(TypedDict):
    query: str
    filter_token: str
    context: str
    sources: List[str]
    grade: str
    answer: str
    retry_count: int


@traceable(name="Query Analysis")
def query_analysis(state: RAGState) -> dict:
    response = model.invoke([
        {"role": "system", "content": QUERY_FILTER_SYSTEM},
        {"role": "user", "content": f"Query: '{state['query']}'. Extract company and year for document filtering."},
    ])
    return {"filter_token": response.content.strip().upper()}


@traceable(name="Document Retrieval")
def retrieve(state: RAGState) -> dict:
    filter_token = state["filter_token"]
    search_kwargs: dict = {"k": 15}

    if filter_token != "NONE":
        search_kwargs["filter"] = Filter(must=[
            FieldCondition(key="metadata.source", match=MatchText(text=filter_token))
        ])

    docs = vectorstore.as_retriever(search_kwargs=search_kwargs).invoke(state["query"])

    if not docs and filter_token != "NONE":
        docs = vectorstore.as_retriever(search_kwargs={"k": 15}).invoke(state["query"])

    context = "\n\n---\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n\n{doc.page_content}"
        for doc in docs
    )
    sources = list(set(doc.metadata.get("source", "Unknown") for doc in docs))

    return {"context": context, "sources": sources}


@traceable(name="Document Grading")
def grade_documents(state: RAGState) -> dict:
    result = (GRADE_PROMPT | model).invoke({
        "question": state["query"],
        "context": state["context"],
    })
    return {"grade": result.content.strip().upper()}


def relax_filter(state: RAGState) -> dict:
    return {
        "filter_token": "NONE",
        "retry_count": state.get("retry_count", 0) + 1,
    }

@traceable(name="Context Reranking")
def rerank(state: RAGState) -> dict:
    query = state["query"]
    chunks = state["context"].split("\n\n---\n\n")

    pairs = [[query, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, chunks), reverse=True)[:10]

    context = "\n\n---\n\n".join(chunk for _, chunk in ranked)
    return {"context": context}

@traceable(name="Final Answer Generation")
def generate(state: RAGState) -> dict:
    response = model.invoke([
        {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Answer the following financial question based ONLY on the provided context.\n\n"
                f"Question: {state['query']}\n\n"
                f"Context:\n{state['context']}"
            ),
        },
    ])
    return {"answer": response.content}


def should_retry(state: RAGState) -> str:
    if "RELEVANT" in state["grade"]:
        return "generate"
    if state.get("retry_count", 0) >= 1:
        return "generate"
    return "retry_retrieve"