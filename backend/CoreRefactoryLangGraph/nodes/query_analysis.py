from langsmith import traceable

from clients import model
from prompts import QUERY_FILTER_SYSTEM
from state import RAGState


@traceable(name="Query Analysis")
def query_analysis(state: RAGState) -> dict:
    response = model.invoke([
        {"role": "system", "content": QUERY_FILTER_SYSTEM},
        {"role": "user", "content": f"Query: '{state['query']}'. Extract company and year for document filtering."},
    ])
    return {"filter_token": response.content.strip().upper()}
