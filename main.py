from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="FinanceBench",
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant for answering questions about financial documents. Use the following context to answer the question. If you don't know the answer, say you don't know.\n\nContext:\n{context}"),
    ("human", "{question}"),
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    query = "What is the year end FY2019 total amount of inventories for Best Buy? Answer in USD millions."

    docs = retriever.invoke(query)
    context = format_docs(docs)

    prompt = prompt_template.format_messages(context=context, question=query)
    response = llm.invoke(prompt)

    print(f"Resposta: {response.content}")
