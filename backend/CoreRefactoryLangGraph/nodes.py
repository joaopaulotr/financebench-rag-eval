from dotenv import load_dotenv
from typing import Any, Dict

from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode
from langsmith import traceable
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from backend.CoreRefactoryLangGraph.tools import retrieve_context

load_dotenv()

tool_node = ToolNode(tools=[retrieve_context])

model = init_chat_model("gpt-4o-mini", model_provider="openai")

@traceable(name="Loopable Retrieval-Augmented Generation")
def run_llm(query: str, system_prompt: str = None) -> Dict[str, Any]:
    if system_prompt is None:
        system_prompt = (
            "You are a financial analyst assistant. Use the retrieved documents to answer the user's query. "
            "If the documents don't contain the answer, say you don't know. Always cite sources."
        )

    agent = create_agent(
        model,
        tools=[retrieve_context],
        state_modifier=system_prompt,
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
