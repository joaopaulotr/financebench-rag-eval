import os
from dotenv import load_dotenv
from typing import Any, Dict
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.agents import create_agent
from qdrant_client import QdrantClient

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="FinanceBench",
)

model = init_chat_model("gpt-3.5-turbo", model_provider="openai")

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant financial documents to help answer the query."""
    # Retrieve top 4 most similar documents
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)

    # Serialize documents for the model
    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )

    # Return both serialized content and raw documents
    return serialized, retrieved_docs

def run_llm(query: str) -> Dict[str, Any]:

    system_prompt = (
        "You are a financial analyst assistant. Use the retrieved documents to answer the user's query. "
        "If the documents don't contain the answer, say you don't know. Always cite sources from the retrieved documents."
    )  

    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

    messages = [{"role": "user", "content": query}]
    
    response = agent.invoke({"messages": messages})

    # Pegando a ultima resposta do modelo, que deve conter a resposta final e os documentos usados (se houver)
    answer = response["messages"][-1].content

    context_docs = []
    for message in response["messages"]:
        artifact = getattr(message, 'artifact', None)
        if artifact:
            context_docs.extend(artifact if isinstance(artifact, list) else list(artifact))

    return {
        "answer": answer,
        "context_docs": context_docs
    }

if __name__ == "__main__":
    result = run_llm("What was JPMorgan's net income in Q1 2021 and which document contains this information?")
    print("Resposta do modelo:")
    print(result["answer"])
    print("\nDocumentos de contexto usados:")
    for doc in result["context_docs"]:
        print(f"  - {doc.metadata.get('source', 'Unknown')}")
