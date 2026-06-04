from langsmith import traceable

from chains.generation import final_answer_chain
from state import RAGState


@traceable(name="Final Answer Generation")
def generate(state: RAGState) -> dict:
    answer = final_answer_chain.invoke({"query": state["query"], "context": state["context"]})
    return {"answer": answer}
