from langchain_community.vectorstores import Chroma
from embedding.embedding_model import load_embedding_model
from config import CHROMA_DB_DIR


def load_vectorstore():

    embedding = load_embedding_model()

    vectordb = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embedding
    )

    return vectordb