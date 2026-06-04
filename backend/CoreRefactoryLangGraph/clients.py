import os

from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse
from qdrant_client import QdrantClient

from sentence_transformers import CrossEncoder

load_dotenv()

model = init_chat_model("gpt-4o", model_provider="openai")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

_qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"), timeout=60
)

vectorstore = QdrantVectorStore(
    client=_qdrant_client,
    collection_name="FinanceBench_v2",
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode="hybrid",
)

reranker = CrossEncoder("BAAI/bge-reranker-base")
