from langsmith import traceable

from state import RAGState


def should_retry(state: RAGState) -> str:
    if state["grade"].strip() == "RELEVANT":
        return "generate"
    if state.get("retry_count", 0) >= 2:
        return "generate"
    return "retry_retrieve"


@traceable(name="Relax Filter")
def relax_filter(state: RAGState) -> dict:
    retry = state.get("retry_count", 0)
    if retry == 0:
        next_filter = state.get("company_filter") or "NONE"
    else:
        next_filter = "NONE"
    return {
        "filter_token": next_filter,
        "retry_count": retry + 1,
    }
