import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader # Carregar PDFs
from langchain_community.text_splitter import CharacterTextSplitter # Chunks de textos
from datasets import load_dataset
load_dotenv()

# Importação de embeddings openAI, input de textos e output seria os embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


loader = DirectoryLoader("data/pdfs", glob="*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()




if __name__ == '__main__':
    print("Ingesting...")
