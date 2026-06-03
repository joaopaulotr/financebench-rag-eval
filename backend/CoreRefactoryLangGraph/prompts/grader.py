from langchain_core.prompts import ChatPromptTemplate

GRADE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a relevance grader for financial document retrieval. "
        "Given a financial question and retrieved document chunks, assess whether the chunks "
        "contain specific financial data (numbers, dates, figures, ratios) "
        "that directly address the question.\n\n"
        "Generic company descriptions do not count.\n\n"
        "Respond with exactly one word: RELEVANT or IRRELEVANT",
    ),
    ("human", "Question: {question}\n\nRetrieved chunks:\n{context}"),
])
