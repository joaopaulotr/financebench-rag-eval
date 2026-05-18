import os
import logging
import time

from dotenv                                         import load_dotenv
from langchain_community.document_loaders           import DirectoryLoader, PyMuPDFLoader
from langchain_openai                               import OpenAIEmbeddings
from langchain_qdrant                               import QdrantVectorStore
from langchain_text_splitters                       import TokenTextSplitter
from qdrant_client                                  import QdrantClient
from qdrant_client.models                           import Distance, VectorParams

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

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
    t0 = time.time()

    logging.info("Carregando PDFs de data/pdfs/...")
    docs = loader.load()
    logging.info(f"PDFs carregados: {len(docs)} páginas em {time.time() - t0:.1f}s")
    logging.info(f"Metadata primeiro doc: {docs[0].metadata}")
    logging.info(f"Prévia conteúdo: {docs[0].page_content[:300]}")

    t1 = time.time()
    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50) # Dividir os textos em pedaços menores para melhor processamento
    text = text_splitter.split_documents(docs) # Gerar embeddings para os textos divididos
    logging.info(f"Chunking concluído: {len(text)} chunks em {time.time() - t1:.1f}s")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Conecta no Qdrant e cria/insere vetores na collection
    collection_name = "FinanceBench"
    qdrant_client = QdrantClient(url="http://localhost:6333")
    logging.info(f"Conectado ao Qdrant em http://localhost:6333")

    existing = [c.name for c in qdrant_client.get_collections().collections]

    BATCH_SIZE = 500
    #Batch indexing para evitar timeout e facilitar reingestão em caso de falha
    def add_batch(vectorstore: QdrantVectorStore, batch: list, batch_number: int, total: int):
        t = time.time()
        try:
            vectorstore.add_documents(batch)
            logging.info(f"  Batch {batch_number + 1}/{total} — {len(batch)} chunks em {time.time() - t:.1f}s")
        except Exception as exc:
            logging.error(f"  Batch {batch_number + 1}/{total} falhou: {exc}")
            return False
        return True

    points_count = qdrant_client.count(collection_name).count if collection_name in existing else 0

    if collection_name in existing and points_count > 0:
        logging.warning(f"Collection '{collection_name}' já existe com {points_count} pontos — pulando ingestão.")
        vectorstore = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            url="http://localhost:6333",
            collection_name=collection_name,
        )
    else:
        try:
            # Cria collection vazia e insere em batches para evitar timeout
            if collection_name not in existing:
                qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
                )
                logging.info(f"Collection '{collection_name}' criada.")
            else:
                logging.warning(f"Collection '{collection_name}' existe mas vazia — reingerindo.")
            vectorstore = QdrantVectorStore(
                client=qdrant_client,
                collection_name=collection_name,
                embedding=embeddings,
            )
            batches = [text[i:i + BATCH_SIZE] for i in range(0, len(text), BATCH_SIZE)]
            logging.info(f"Iniciando ingestão: {len(text)} chunks em {len(batches)} batches de {BATCH_SIZE}...")
            t2 = time.time()
            successful = sum(
                add_batch(vectorstore, batch, idx, len(batches)) for idx, batch in enumerate(batches)
            )
            logging.info(f"Ingestão concluída: {successful}/{len(batches)} batches em {time.time() - t2:.1f}s")
        except Exception as exc:
            logging.error(f"Erro na ingestao: {exc}")
            raise

    logging.info(f"Pipeline total: {time.time() - t0:.1f}s")
    logging.info(f"Collections no Qdrant: {[c.name for c in qdrant_client.get_collections().collections]}")