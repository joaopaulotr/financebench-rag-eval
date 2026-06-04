from langsmith import traceable

from chains.query_analysis import query_filter_chain
from state import RAGState


@traceable(name="Query Analysis")
def query_analysis(state: RAGState) -> dict:
    result = query_filter_chain.invoke({"query": state["query"]})
    return {
        "filter_token": result.filter_token.strip().upper(),
        "company_filter": result.company_filter.strip().upper(),
    }
