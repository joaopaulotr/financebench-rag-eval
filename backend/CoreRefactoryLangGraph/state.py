from typing import TypedDict, List


class RAGState(TypedDict):
    query: str
    filter_token: str
    context: str
    sources: List[str]
    grade: str
    answer: str
    retry_count: int
