from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from vectorstore.chroma_db import load_vectorstore
from config import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from embedding.embedding_model import load_embedding_model
from langchain_community.vectorstores import Chroma


def ingest_pdf():

    loader = PyMuPDFLoader(DATA_PATH)

    documents = loader.load()

    splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)

    embedding = load_embedding_model()

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory="vectorstore/db"
    )

    vectordb.persist()

    print("Vector DB built successfully")


if __name__ == "__main__":
    ingest_pdf()