from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from clients import model
from prompts import QUERY_FILTER_SYSTEM


class QueryFilter(BaseModel):
    filter_token: str = Field(description="Company + year token, e.g. ADOBE_2022 or NONE")
    company_filter: str = Field(description="Company name only, e.g. ADOBE or NONE")


_prompt = ChatPromptTemplate.from_messages([
    ("system", QUERY_FILTER_SYSTEM),
    ("human", "Query: '{query}'"),
])

query_filter_chain = _prompt | model.with_structured_output(QueryFilter)
