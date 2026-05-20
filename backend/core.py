import os
from dotenv import load_dotenv
from typing import Any, Dict

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse
from langchain_openai import OpenAIEmbeddings
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25") # Inicializa modelo léxico

vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    sparse_embedding=sparse_embeddings, # Habilita vetores esparsos (BM25)
    url="http://localhost:6333",
    collection_name="FinanceBench",
    retrieval_mode="hybrid"             # LangChain aplica RRF (fusão) automaticamente
)

model = init_chat_model("gpt-4o-mini", model_provider="openai")
MAX_ITERATIONS = 3  # O agente só poderá acionar ferramentas 3 vezes por query

@tool
def retrieve_context(query: str):
    """Retrieve relevant financial documents to help answer the query."""
    
    response_metadata_filter = model.invoke({
        "messages": [
            {
                "role": "system", 
                "content": (
                    "Extract the company name from the financial question. "
                    "Return ONLY the company name as it appears in SEC filing filenames "
                    "(e.g. JOHNSON_JOHNSON, ADOBE, AMD, 3M, AMCOR). "
                    "If no specific company is mentioned, return: NONE."
                )
            },
            {"role": "user", "content": f"Here is the user's query: '{query}'. Extract the company name for filtering."}
        ]
    })

    company = response_metadata_filter.content.strip().upper()

    search_kwargs = {"k": 6}
    if company != "NONE":
        search_kwargs["filter"] = {"company": company} 

    retrieved_docs = vectorstore.as_retriever(search_kwargs=search_kwargs).invoke(query)

    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )

    sources_used = [doc.metadata.get('source', 'Unknown') for doc in retrieved_docs]
    return {"context": serialized, "sources": sources_used}

@traceable(name="Loopable Retrieval-Augmented Generation")
def run_llm(query: str, system_prompt: str = None) -> Dict[str, Any]:
    if system_prompt is None:
        system_prompt = (
            "You are a financial analyst assistant. Use the retrieved documents to answer the user's query. "
            "If the documents don't contain the answer, say you don't know. Always cite sources."
        )  

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}"), 
    ])

    agent = create_tool_calling_agent(model, tools=[retrieve_context], prompt=prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=[retrieve_context],
        max_iterations=MAX_ITERATIONS,    # Impede que ele entre em loops infinitos
        early_stopping_method="force", 
        return_intermediate_steps=True    
    )
    
    response = agent_executor.invoke({"input": query})

    final_response = response["output"]

    if final_response == "Agent stopped due to iteration limit or time limit.":
        final_response = "Aviso: O agente atingiu o limite máximo de iterações e a operação foi interrompida para evitar loops."

    # Extraindo as fontes do log de execução
    # Extraindo as fontes do log de execução das ferramentas
    context_docs = []
    for action, observation in response.get("intermediate_steps", []):
        if action.tool == "retrieve_context":
            if isinstance(observation, dict) and "sources" in observation:
                context_docs.extend(observation["sources"])

    # Remove duplicatas caso a ferramenta tenha sido acionada múltiplas vezes
    context_docs = list(set(context_docs))

    return {
        "answer": final_response,
        "context_docs": context_docs
    }

if __name__ == "__main__":
    result = run_llm("What was JPMorgan's net income in Q1 2021 and which document contains this information?")
    print("Resposta do modelo:\n", result["answer"])
    print("\nDocumentos de contexto usados:")
    for source in result["context_docs"]:
        print(f"  - {source}")