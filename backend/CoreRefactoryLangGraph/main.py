from dotenv import load_dotenv
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import MessagesState, StateGraph, END
from langsmith import traceable

from nodes import run_agent_reasoning, tool_node

load_dotenv()

AGENT_REASON = "agent_reason"
ACT = "retrieve_context"
LAST = -1


def should_continue(state: MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT


flow = StateGraph(MessagesState)
flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)
flow.add_conditional_edges(AGENT_REASON, should_continue, {END: END, ACT: ACT})
flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")

DEFAULT_SYSTEM_PROMPT = (
    "You are a financial analyst assistant. Use the retrieved documents to answer the user's query. "
    "If the documents don't contain the answer, say you don't know. Always cite sources."
)


@traceable(name="Loopable Retrieval-Augmented Generation")
def run_llm(query: str, system_prompt: str = None) -> Dict[str, Any]:
    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    messages = [SystemMessage(content=sys_prompt), HumanMessage(content=query)]

    try:
        response = app.invoke(
            {"messages": messages},
            config={"recursion_limit": 10},
        )
        final_answer = response["messages"][LAST].content
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
