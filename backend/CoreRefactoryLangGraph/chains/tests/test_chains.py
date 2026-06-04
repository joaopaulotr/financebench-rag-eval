from dotenv import load_dotenv

load_dotenv()

from chains.grade import grade_chain, DocumentGrade
from chains.query_analysis import query_filter_chain, QueryFilter
from chains.generation import final_answer_chain
from clients import vectorstore


def test_query_analysis_extracts_company_and_year() -> None:
    result: QueryFilter = query_filter_chain.invoke(
        {"query": "What is the FY2018 capital expenditure for 3M?"}
    )
    assert "3M" in result.filter_token
    assert "2018" in result.filter_token
    assert result.company_filter == "3M"


def test_query_analysis_extracts_company_without_year() -> None:
    result: QueryFilter = query_filter_chain.invoke(
        {"query": "What is Adobe's gross margin?"}
    )
    assert "ADOBE" in result.filter_token
    assert result.company_filter == "ADOBE"


def test_query_analysis_returns_none_for_no_company() -> None:
    result: QueryFilter = query_filter_chain.invoke(
        {"query": "What is the definition of EBITDA?"}
    )
    assert result.filter_token == "NONE"
    assert result.company_filter == "NONE"


def test_grade_relevant() -> None:
    question = "What was Adobe's revenue in FY2022?"
    docs = vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    result: DocumentGrade = grade_chain.invoke({"question": question, "context": context})
    assert result.grade == "RELEVANT"


def test_grade_irrelevant() -> None:
    question = "How to make pizza at home?"
    context = "The capital expenditure for 3M in FY2018 was $1,577 million as shown in the cash flow statement."

    result: DocumentGrade = grade_chain.invoke({"question": question, "context": context})
    assert result.grade == "IRRELEVANT"


def test_generation_chain() -> None:
    question = "What was Adobe's FY2022 revenue?"
    context = (
        "Source: ADOBE_2022_10K\n\n"
        "Adobe Inc. reported total revenue of $17.606 billion for fiscal year 2022, "
        "representing a 12% increase compared to fiscal year 2021."
    )

    answer: str = final_answer_chain.invoke({"query": question, "context": context})
    assert isinstance(answer, str)
    assert len(answer) > 0
    assert "17" in answer or "Adobe" in answer
