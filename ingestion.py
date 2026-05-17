import os

from dotenv                                         import load_dotenv
from datasets                                       import load_dataset
from langchain_community.document_loaders           import DirectoryLoader, PyMuPDFLoader
from langchain_openai                               import OpenAIEmbeddings
from langchain_qdrant                               import QdrantVectorStore
from langchain_text_splitters                       import TokenTextSplitter
from qdrant_client                                  import QdrantClient

load_dotenv()

# Carrega PDFs do diretorio e extrai texto/paginas
loader = DirectoryLoader(
    "data/pdfs",
    glob="*.pdf",
    loader_cls=PyMuPDFLoader,
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

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
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