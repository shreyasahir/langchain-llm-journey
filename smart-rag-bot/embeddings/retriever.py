from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()
chunk_size = int(os.getenv("CHUNK_SIZE", 500))
chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 50))
data_path = os.getenv("DATA_FILE", "../data/langgraph_doc.txt")

class DocumentRetriever:
    def __init__(self, filepath: str = data_path, model: str = "nomic-embed-text"):
        self.filepath = filepath  # ✅ do NOT override with `data_path`
        self.model = model
        self.vectorstore = self._load_and_embed()

    def _load_and_embed(self) -> FAISS:
        # Load the file
        loader = TextLoader(self.filepath)
        documents = loader.load()

        # Split the documents
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks: List[Document] = splitter.split_documents(documents)

        # Embed using Ollama
        embeddings = OllamaEmbeddings(model=self.model)

        # Create vector store
        return FAISS.from_documents(chunks, embeddings)

    def search(self, query: str, k: int = 3) -> List[str]:
        results = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]
