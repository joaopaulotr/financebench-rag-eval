from graph import app


def run(query: str) -> dict:
    result = app.invoke({
        "query": query,
        "filter_token": "",
        "context": "",
        "sources": [],
        "grade": "",
        "answer": "",
        "retry_count": 0,
    })
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "context": result["context"],
    }


if __name__ == "__main__":
    query = (
        "What is the FY2018 capital expenditure amount (in USD millions) for 3M? "
        "Give a response to the question by relying on the details shown in the cash flow statement."
    )
    # Expected Answer: $1577.00M
    result = run(query)
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources: {result['sources']}")
