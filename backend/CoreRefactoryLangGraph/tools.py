from dotenv import                      load_dotenv

from langchain.chat_models import       init_chat_model
from langchain_openai import            OpenAIEmbeddings
from langchain_qdrant import            QdrantVectorStore, FastEmbedSparse

from sentence_transformers import CrossEncoder

load_dotenv()

model = init_chat_model("gpt-4o-mini", model_provider="openai")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    url="http://localhost:6333",
    collection_name="FinanceBench",
    retrieval_mode="hybrid",
)

reranker = CrossEncoder("BAAI/bge-reranker-base")