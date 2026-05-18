import os

from dotenv                                         import load_dotenv
from langchain_community.document_loaders           import DirectoryLoader, PyMuPDFLoader
from langchain_openai                               import OpenAIEmbeddings
from langchain_qdrant                               import QdrantVectorStore
from langchain_text_splitters                       import TokenTextSplitter
from qdrant_client                                  import QdrantClient
from qdrant_client.models                           import Distance, VectorParams

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

    BATCH_SIZE = 500

    def add_batch(vectorstore: QdrantVectorStore, batch: list, batch_number: int):
        try:
            vectorstore.add_documents(batch)
            print(f"Batch {batch_number} adicionado com sucesso.")
        except Exception as exc:
            print(f"Erro na ingestao do batch {batch_number}: {exc}")
            return False
        return True

    if collection_name in existing:
        print("Collection já existe, pulando ingestão.")
        vectorstore = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            url="http://localhost:6333",
            collection_name=collection_name,
        )
    else:
        try:
            # Cria collection vazia e insere em batches para evitar timeout
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
            vectorstore = QdrantVectorStore(
                client=qdrant_client,
                collection_name=collection_name,
                embedding=embeddings,
            )
            batches = [text[i:i + BATCH_SIZE] for i in range(0, len(text), BATCH_SIZE)]
            successful_batches = sum(
                add_batch(vectorstore, batch, idx) for idx, batch in enumerate(batches)
            )
            print(f"{successful_batches} batches adicionados com sucesso de {len(batches)}.")
        except Exception as exc:
            print(f"Erro na ingestao: {exc}")
            raise

    print("Ingestão concluída no Qdrant.")
    print(qdrant_client.get_collections())