import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "shenshen.pdf")

CHROMA_DB_DIR = os.path.join(BASE_DIR, "vectorstore", "db")

LLM_MODEL_PATH = "/root/Agent/internship_demo/RAG/qwen2.5-3b-instruct-q5_k_m.gguf"

EMBEDDING_MODEL = "/root/Agent/internship_demo/RAG/models/bge-small-zh-v1.5"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100