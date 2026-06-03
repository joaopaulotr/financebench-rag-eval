from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from tools import model, retrieve_context

model_with_tools = model.bind_tools([retrieve_context])

tool_node = ToolNode(tools=[retrieve_context])


def run_agent_reasoning(state: MessagesState):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}
