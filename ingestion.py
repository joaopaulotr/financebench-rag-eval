import os

from dotenv                                         import load_dotenv
from datasets                                       import load_dataset
from langchain_community.document_loaders           import DirectoryLoader, PyPDFLoader # Carregar PDFs
from langchain_openai                               import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant                               import QdrantVectorStore
from langchain_text_splitters                       import TokenTextSplitter
from qdrant_client                                  import QdrantClient

load_dotenv()

# Carrega PDFs do diretorio e extrai texto/paginas
loader = DirectoryLoader(
    "data/pdfs",
    glob="*.pdf",
    loader_cls=PyPDFLoader,
    silent_errors=True,
    show_progress=True,
)# Carregando diretório de PDFs, usando o PyPDFLoader para ler os arquivos PDF

#loader = PyPDFLoader("data/pdfs/" + "JPMORGAN_2021Q1_10Q.pdf") # Carregando um arquivo PDF específico, usando o PyPDFLoader para ler o arquivo PDF

if __name__ == '__main__':
    docs = loader.load()
    print(f"Páginas totais: {len(docs)}")
    print(f"Metadata primeiro doc: {docs[0].metadata}")
    print(f"Prévia conteúdo: {docs[0].page_content[:300]}")

    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50) # Dividir os textos em pedaços menores para melhor processamento
    text = text_splitter.split_documents(docs) # Gerar embeddings para os textos divididos
    print(f"Gerados {len(text)} chunks de texto.")

    embeddings = OpenAIEmbeddings() # Gerar embeddings usando OpenAI
    print("ingesting...")
    print(f"Chunks para ingestao: {len(text)}")

    # Conecta no Qdrant e cria/insere vetores na collection
    collection_name = "FinanceBench"
    qdrant_client = QdrantClient(url="http://localhost:6333")


    existing = [c.name for c in qdrant_client.get_collections().collections]

    if collection_name in existing:
        print("Collection já existe, pulando ingestão.")
        vectorstore = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            url="http://localhost:6333",
            collection_name=collection_name,
        )
    else:
        try:
            vectorstore = QdrantVectorStore.from_documents(
                documents=text,
                embedding=embeddings,
                url="http://localhost:6333",
                collection_name=collection_name,
            )
        except Exception as exc:
            print(f"Erro na ingestao: {exc}")
            raise
    print("Ingestão concluída no Qdrant.")
    print(qdrant_client.get_collections())

    # Faz busca por similaridade e retorna os k chunks mais proximos
    results = vectorstore.similarity_search(
        "What is the year end FY2019 total amount of inventories for Best Buy? Answer in USD millions.",
        k=3,
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    context = "\n\n".join(result.page_content for result in results)
    prompt = [
        (
            "system",
            f"You are a helpful assistant for answering questions about financial documents. Use the following context to answer the question. If you don't know the answer, say you don't know.\n\nContext:\n{context}",
        ),
        (
            "human",
            "What is the year end FY2019 total amount of inventories for Best Buy? Answer in USD millions.",
        ),
    ]

    model_response = llm.invoke(prompt)
    print(f"Resposta do modelo: {model_response}")