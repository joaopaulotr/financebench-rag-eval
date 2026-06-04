from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from clients import model
from prompts import ANALYST_SYSTEM_PROMPT

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ANALYST_SYSTEM_PROMPT),
        (
            "human",
            "Answer the following financial question based ONLY on the provided context.\n\nQuestion: {query}\n\nContext:\n{context}",
        ),
    ]
)

final_answer_chain = _prompt | model | StrOutputParser()
