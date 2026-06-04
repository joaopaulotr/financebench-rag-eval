import os
import logging
import time

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyMuPDFLoader,
    BSHTMLLoader,
)
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse
from langchain_text_splitters import TokenTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

pdf_loader = DirectoryLoader(
    "data/pdfs",
    glob="*.pdf",
    loader_cls=PyMuPDFLoader,
    silent_errors=False,
    show_progress=True,
)

htm_loader = DirectoryLoader(
    "data/_tmp_htm",
    glob="*.htm",
    loader_cls=BSHTMLLoader,
    silent_errors=False,
    show_progress=True,
    loader_kwargs={"open_encoding": "utf-8"},
)

if __name__ == "__main__":
    t0 = time.time()

    logging.info("Carregando PDFs de data/pdfs/...")
    pdf_docs = pdf_loader.load()
    logging.info(f"PDFs carregados: {len(pdf_docs)} páginas em {time.time() - t0:.1f}s")

    logging.info("Carregando HTMs de data/_tmp_htm/...")
    htm_docs = htm_loader.load()
    logging.info(f"HTMs carregados: {len(htm_docs)} páginas")

    docs = pdf_docs + htm_docs
    logging.info(f"Total documentos: {len(docs)}")
    if docs:
        logging.info(f"Metadata primeiro doc: {docs[0].metadata}")
        logging.info(f"Prévia conteúdo: {docs[0].page_content[:300]}")

    t1 = time.time()
    text_splitter = TokenTextSplitter(
        chunk_size=512, chunk_overlap=50
    )  # Dividir os textos em pedaços menores para melhor processamento
    text = text_splitter.split_documents(docs)
    logging.info(f"Chunking concluído: {len(text)} chunks em {time.time() - t1:.1f}s")

    def add_context_prefix(doc):
        source = doc.metadata.get("source", "")
        filename = (
            source.replace("\\", "/")
            .split("/")[-1]
            .replace(".pdf", "")
            .replace(".htm", "")
        )
        parts = filename.split("_")

        company = " ".join(
            p
            for p in parts
            if not p.isdigit()
            and p not in ("10K", "10Q", "8K", "EARNINGS", "Q1", "Q2", "Q3", "Q4")
        )
        year = next((p for p in parts if len(p) == 4 and p.isdigit()), "Unknown")
        doc_type = next(
            (p for p in parts if p in ("10K", "10Q", "8K", "EARNINGS")), "Filing"
        )
        quarter = next((p for p in parts if p in ("Q1", "Q2", "Q3", "Q4")), "")

        prefix = f"Company: {company} | Document: {doc_type} | Year: {year}"
        if quarter:
            prefix += f" {quarter}"

        doc.page_content = f"{prefix}\n\n{doc.page_content}"
        return doc

    text = [add_context_prefix(chunk) for chunk in text]
    logging.info(f"Prefixo contextual aplicado. Exemplo: {text[0].page_content[:120]}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    # Conecta no Qdrant e cria/insere vetores na collection
    collection_name = "FinanceBench_v2"
    qdrant_client = QdrantClient(url="http://localhost:6333")
    logging.info(f"Conectado ao Qdrant em http://localhost:6333")

    existing = [c.name for c in qdrant_client.get_collections().collections]

    BATCH_SIZE = 500

    # Batch indexing para evitar timeout e facilitar reingestão em caso de falha
    def add_batch(
        vectorstore: QdrantVectorStore, batch: list, batch_number: int, total: int
    ):
        t = time.time()
        try:
            vectorstore.add_documents(batch)
            logging.info(
                f"  Batch {batch_number + 1}/{total} — {len(batch)} chunks em {time.time() - t:.1f}s"
            )
        except Exception as exc:
            logging.error(f"  Batch {batch_number + 1}/{total} falhou: {exc}")
            return False
        return True

    points_count = (
        qdrant_client.count(collection_name).count if collection_name in existing else 0
    )

    if collection_name in existing and points_count > 0:
        logging.warning(
            f"Collection '{collection_name}' já existe com {points_count} pontos — pulando ingestão."
        )

        # Se a coleção já existe, conecta ativando os dois modelos e a busca híbrida
        vectorstore = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            sparse_embedding=sparse_embeddings,
            url="http://localhost:6333",
            collection_name=collection_name,
            retrieval_mode="hybrid",
        )
    else:
        try:
            # Cria collection vazia
            if collection_name not in existing:
                qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
                    sparse_vectors_config={"langchain-sparse": SparseVectorParams()},
                )
                logging.info(f"Collection '{collection_name}' criada.")
            else:
                logging.warning(
                    f"Collection '{collection_name}' existe mas vazia — reingerindo."
                )

            # Instancia o VectorStore passando os DOIS modelos (denso e esparso)
            vectorstore = QdrantVectorStore(
                client=qdrant_client,
                collection_name=collection_name,
                embedding=embeddings,
                sparse_embedding=sparse_embeddings,
                retrieval_mode="hybrid",
            )

            batches = [
                text[i : i + BATCH_SIZE] for i in range(0, len(text), BATCH_SIZE)
            ]
            logging.info(
                f"Iniciando ingestão: {len(text)} chunks em {len(batches)} batches de {BATCH_SIZE}..."
            )
            t2 = time.time()
            successful = sum(
                add_batch(vectorstore, batch, idx, len(batches))
                for idx, batch in enumerate(batches)
            )
            logging.info(
                f"Ingestão concluída: {successful}/{len(batches)} batches em {time.time() - t2:.1f}s"
            )
        except Exception as exc:
            logging.error(f"Erro na ingestao: {exc}")
            raise

    logging.info(f"Pipeline total: {time.time() - t0:.1f}s")
    logging.info(
        f"Collections no Qdrant: {[c.name for c in qdrant_client.get_collections().collections]}"
    )

    # Verificação: todos os PDFs tem chunks no Qdrant?
    logging.info("=== VERIFICAÇÃO DE COBERTURA ===")
    pdf_files = {
        f.replace(".pdf", "") for f in os.listdir("data/pdfs") if f.endswith(".pdf")
    } | {
        f.replace(".htm", "") for f in os.listdir("data/_tmp_htm") if f.endswith(".htm")
    }
    sources_indexed = set()
    offset = None
    while True:
        result = qdrant_client.scroll(
            collection_name=collection_name,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in result[0]:
            src = point.payload.get("metadata", {}).get("source", "")
            filename = (
                src.replace("\\", "/")
                .split("/")[-1]
                .replace(".pdf", "")
                .replace(".htm", "")
            )
            sources_indexed.add(filename)
        offset = result[1]
        if offset is None:
            break

    missing = pdf_files - sources_indexed
    logging.info(f"PDFs no disco: {len(pdf_files)}")
    logging.info(f"PDFs indexados: {len(sources_indexed)}")
    if missing:
        logging.warning(f"PDFs FALTANDO no Qdrant ({len(missing)}):")
        for m in sorted(missing):
            logging.warning(f"  - {m}")
    else:
        logging.info("✓ Todos os PDFs estão indexados.")
