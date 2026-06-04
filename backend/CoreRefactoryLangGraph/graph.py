from langgraph.graph import StateGraph, END

from state import RAGState
from nodes import (
    query_analysis,
    retrieve,
    grade_documents,
    relax_filter,
    generate,
    should_retry,
    rerank,
)


def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("query_analysis", query_analysis)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("relax_filter", relax_filter)
    graph.add_node("generate", generate)
    graph.add_node("rerank", rerank)

    graph.set_entry_point("query_analysis")
    graph.add_edge("query_analysis", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "grade_documents")

    graph.add_conditional_edges(
        "grade_documents",
        should_retry,
        {
            "generate": "generate",
            "retry_retrieve": "relax_filter",
        },
    )
    graph.add_edge("relax_filter", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()


app = build_graph()

# app.get_graph().draw_mermaid_png(output_file_path="flowv3.png")
