from dotenv import load_dotenv

load_dotenv()

from CoreRefactoryLangGraph.chains.grade import grade_chain, DocumentGrade
from CoreRefactoryLangGraph.nodes.retrieve import retrieve

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