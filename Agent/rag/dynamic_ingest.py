import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from llm.llm_loader import load_embedding_model
from config import CHROMA_DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP

def process_uploaded_file(file_path: str):
    """解析上传的 PDF 并存入 ChromaDB"""
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)

    embedding = load_embedding_model()
    
    # 存入向量数据库
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=CHROMA_DB_DIR
    )
    vectordb.persist()
    
    return chunks # 返回 chunks 用于后续初始化 BM25