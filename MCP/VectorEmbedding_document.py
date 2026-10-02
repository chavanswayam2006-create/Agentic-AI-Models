from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma 

#1. Initialize local embeddings model via ollama 
embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

#2. Store embeddings locally on disk
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Vector database created and persisted locally.")