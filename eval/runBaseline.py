import csv
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

load_dotenv()

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="FinanceBench",
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant for answering questions about financial documents. Use the following context to answer the question. If you don't know the answer, say you don't know.\n\nContext:\n{context}"),
    ("human", "{question}"),
])


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = (
    RunnablePassthrough.assign(
        context=itemgetter("question") | retriever | format_docs
    ) | prompt_template | llm | StrOutputParser()
)

rows = []
with open("eval/baseline_20.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

for i, row in enumerate(rows):
    print(f"[{i+1}/20] {row['question'][:80]}...")
    answer = chain.invoke({"question": row["question"]})
    row["model_answer"] = answer
    print(f"  -> {answer[:100]}")

with open("eval/baseline_20.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
