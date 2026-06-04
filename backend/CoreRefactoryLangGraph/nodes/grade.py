from langsmith import traceable

from tools import model
from prompts import GRADE_PROMPT
from state import RAGState
from pydantic import BaseModel, Field

class DocumentGrade(BaseModel):
    grade: str = Field(description="The grade assigned to the retrieved documents, RELEVANT or IRRELEVANT")

structured_grade = model.with_structured_output(DocumentGrade)

retrival_grader = GRADE_PROMPT | structured_grade

@traceable(name="Document Grading")
def grade_documents(state: RAGState) -> dict:
    result = retrival_grader.invoke({
        "question": state["query"],
        "context": state["context"],
    })
    return {"grade": result.grade}
