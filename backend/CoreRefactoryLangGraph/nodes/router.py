from langsmith import traceable

from state import RAGState


def should_retry(state: RAGState) -> str:
    if "RELEVANT" in state["grade"]:
        return "generate"
    if state.get("retry_count", 0) >= 1:
        return "generate"
    return "retry_retrieve"


@traceable(name="Relax Filter")
def relax_filter(state: RAGState) -> dict:
    return {
        "filter_token": "NONE",
        "retry_count": state.get("retry_count", 0) + 1,
    }
