from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="FinanceBench",
)

# Testei com 3 e acabou em alguns casos não achando pela vastidão de documentos, aumentando para 6 e achei que ficou mais robusto, mesmo que com um pouco mais de ruído
# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for answering questions about financial documents. Use the following context to answer the question. If you don't know the answer, say you don't know.\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_retrieval_chain_with_lcel():  # Chain de recuperação usando LCEL (LangChain Embeddings + LLM)
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


if __name__ == "__main__":
    query = "What is the year end FY2019 total amount of inventories for Best Buy? Answer in USD millions."

    chain = create_retrieval_chain_with_lcel()
    result = chain.invoke({"question": query})

    print(f"Resposta: {result}")
