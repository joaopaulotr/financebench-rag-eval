from dotenv import load_dotenv
from typing import Any, Dict, Optional

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse
from langchain_openai import OpenAIEmbeddings
from langsmith import traceable
from qdrant_client.models import Filter, FieldCondition, MatchText

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    url="http://localhost:6333",
    collection_name="FinanceBench",
    retrieval_mode="hybrid",
)

model = init_chat_model("gpt-4o-mini", model_provider="openai")


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant financial documents to help answer the query."""
    response_filter = model.invoke(
        [
            {
                "role": "system",
                "content": (
                    "Extract the company name and fiscal year from the financial question. "
                    "Return ONLY a filter token in this format: COMPANY_YEAR (e.g. ADOBE_2022, AMD_2015, JOHNSON_JOHNSON_2022). "
                    "Use the company name exactly as it appears in SEC filing filenames "
                    "(e.g. JOHNSON_JOHNSON, ADOBE, AMD, 3M, AMCOR, ACTIVISIONBLIZZARD, KRAFTHEINZ, MGMRESORTS). "
                    "If the year is not mentioned or is ambiguous, return only the company name (e.g. ADOBE). "
                    "If no specific company is mentioned, return: NONE."
                ),
            },
            {
                "role": "user",
                "content": f"Query: '{query}'. Extract company and year for document filtering.",
            },
        ]
    )

    filter_token = response_filter.content.strip().upper()

    search_kwargs: dict = {"k": 6}
    if filter_token != "NONE":
        search_kwargs["filter"] = Filter(
            must=[
                FieldCondition(
                    key="metadata.source", match=MatchText(text=filter_token)
                )
            ]
        )

    retrieved_docs = vectorstore.as_retriever(search_kwargs=search_kwargs).invoke(query)

    if not retrieved_docs and filter_token != "NONE":
        retrieved_docs = vectorstore.as_retriever(search_kwargs={"k": 6}).invoke(query)

    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    sources = [doc.metadata.get("source", "Unknown") for doc in retrieved_docs]

    return serialized, {"sources": sources}


@traceable(name="Loopable Retrieval-Augmented Generation")
def run_llm(query: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    if system_prompt is None:
        system_prompt = (
            "You are a financial analyst assistant. Use the retrieved documents to answer the user's query. "
            "If the documents don't contain the answer, say you don't know. Always cite sources."
        )

    agent = create_agent(
        model,
        tools=[retrieve_context],
        system_prompt=system_prompt,
    )

    try:
        response = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"recursion_limit": 10},
        )
        final_answer = response["messages"][-1].content
    except Exception:
        final_answer = "Aviso: O agente atingiu o limite maximo de iteracoes."
        response = {"messages": []}

    context_docs = []
    context_text_parts = []
    for msg in response["messages"]:
        if isinstance(msg, ToolMessage) and msg.name == "retrieve_context":
            if msg.artifact and isinstance(msg.artifact, dict):
                context_docs.extend(msg.artifact.get("sources", []))
            if msg.content:
                context_text_parts.append(msg.content)

    context_docs = list(set(context_docs))
    context_text = "\n\n---\n\n".join(context_text_parts)

    return {
        "answer": final_answer,
        "context_docs": context_docs,
        "context_text": context_text,
    }


if __name__ == "__main__":
    result = run_llm("What was JPMorgan's net income in Q1 2021?")
    print("Resposta:\n", result["answer"])
    print("\nFontes:")
    for s in result["context_docs"]:
        print(f"  - {s}")
