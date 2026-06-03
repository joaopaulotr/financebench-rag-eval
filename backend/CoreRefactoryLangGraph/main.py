from langgraph.graph import StateGraph, END

from nodes import (
    RAGState,
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

    graph.add_conditional_edges("grade_documents", should_retry, {
        "generate": "generate",
        "retry_retrieve": "relax_filter",
    })
    graph.add_edge("relax_filter", "retrieve")
    graph.add_edge("generate", END)
    
    return graph.compile()


app = build_graph()

#Graph visualization
app.get_graph().draw_mermaid_png(output_file_path="flowv3.png")

def run(query: str) -> dict:
    result = app.invoke({
        "query": query,
        "filter_token": "",
        "context": "",
        "sources": [],
        "grade": "",
        "answer": "",
        "retry_count": 0,
    })
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "context": result["context"],
    }


if __name__ == "__main__":
    query = (
        "What is the FY2018 capital expenditure amount (in USD millions) for 3M? "
        "Give a response to the question by relying on the details shown in the cash flow statement."
    )
    # Expected Answer: "$1577.00M"
    result = run(query)
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources: {result['sources']}")