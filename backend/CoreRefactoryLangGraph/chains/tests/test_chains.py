from dotenv import load_dotenv

load_dotenv()

from chains.grade import grade_chain, DocumentGrade
from chains.query_analysis import query_filter_chain, QueryFilter
from chains.generation import final_answer_chain

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


def test_retrieval_grade_answer_relevant() -> None:
    question = "What was Adobe's revenue in 2022?"
    docs = retrieve.invoke(question)
    docs_txt = docs[1].page_content


    res: DocumentGrade = grade_chain.invoke(
        {
            "question": question,
            "retrieved_docs": docs_txt,
        }
    )

    assert res.grade == "RELEVANT"

def test_retrieval_grade_answer_irrelevant() -> None:
    question = "How to make pizza at home?"
    docs = retrieve.invoke(question)
    docs_txt = docs[1].page_content


    res: DocumentGrade = grade_chain.invoke(
        {
            "question": question,
            "retrieved_docs": docs_txt,
        }
    )

    assert res.grade == "RELEVANT"

def test_generation_chain() -> None:
    question = "What was Adobe's revenue in 2022?"
    docs = retrieve.invoke(question)
    
    generation = final_answer_chain.invoke(
        {
            "query": question,
            "context": docs[1].page_content,
        }
    )
    print(generation)
    assert "Adobe's revenue in 2022" in generation