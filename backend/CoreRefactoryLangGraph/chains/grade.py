from pydantic import BaseModel, Field

from clients import model
from prompts import GRADE_PROMPT


class DocumentGrade(BaseModel):
    grade: str = Field(description="The grade assigned to the retrieved documents, RELEVANT or IRRELEVANT")


grade_chain = GRADE_PROMPT | model.with_structured_output(DocumentGrade)
