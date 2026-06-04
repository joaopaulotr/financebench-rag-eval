from langsmith import traceable

from tools import model
from prompts import ANALYST_SYSTEM_PROMPT
from state import RAGState


@traceable(name="Final Answer Generation")
def generate(state: RAGState) -> dict:
    response = model.invoke([
        {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Answer the following financial question based ONLY on the provided context.\n\n"
                f"Question: {state['query']}\n\n"
                f"Context:\n{state['context']}"
            ),
        },
    ])
    return {"answer": response.content}
