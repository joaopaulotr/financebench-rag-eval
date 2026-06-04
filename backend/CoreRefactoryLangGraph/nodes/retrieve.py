from langsmith import traceable
from qdrant_client.models import Filter, FieldCondition, MatchText

from clients import vectorstore
from state import RAGState


@traceable(name="Document Retrieval")
def retrieve(state: RAGState) -> dict:
    filter_token = state["filter_token"]
    search_kwargs: dict = {"k": 8}

    if filter_token != "NONE":
        search_kwargs["filter"] = Filter(must=[
            FieldCondition(key="metadata.source", match=MatchText(text=filter_token))
        ])

    docs = vectorstore.as_retriever(search_kwargs=search_kwargs).invoke(state["query"])

    if not docs and filter_token != "NONE":
        docs = vectorstore.as_retriever(search_kwargs={"k": 8}).invoke(state["query"])

    context = "\n\n---\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n\n{doc.page_content}"
        for doc in docs
    )
    sources = list(set(doc.metadata.get("source", "Unknown") for doc in docs))

    return {"context": context, "sources": sources}
