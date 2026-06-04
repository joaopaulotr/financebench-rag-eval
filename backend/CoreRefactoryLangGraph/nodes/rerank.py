from langsmith import traceable

from tools import reranker
from state import RAGState


@traceable(name="Context Reranking")
def rerank(state: RAGState) -> dict:
    query = state["query"]
    chunks = [c for c in state["context"].split("\n\n---\n\n") if c.strip()]

    if not chunks:
        return {"context": ""}

    pairs = [[query, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, chunks), reverse=True)[:10]

    context = "\n\n---\n\n".join(chunk for _, chunk in ranked)
    return {"context": context}
