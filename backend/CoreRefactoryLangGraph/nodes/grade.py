from langsmith import traceable

from chains.grade import grade_chain
from state import RAGState


@traceable(name="Document Grading")
def grade_documents(state: RAGState) -> dict:
    result = grade_chain.invoke(
        {
            "question": state["query"],
            "context": state["context"],
        }
    )
    return {"grade": result.grade}
