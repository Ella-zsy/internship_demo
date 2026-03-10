import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 动态上传的临时文件目录
UPLOAD_DIR = os.path.join(BASE_DIR, "temp_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 向量数据库目录
CHROMA_DB_DIR = os.path.join(BASE_DIR, "vectorstore", "db")

# 模型配置 (替换为你的本地实际路径)
LLM_MODEL_PATH = "/root/Agent/internship_demo/Agent/qwen2.5-3b-instruct-q5_k_m.gguf"
EMBEDDING_MODEL = "/root/Agent/internship_demo/Agent/models/bge-small-zh-v1.5"

# RAG Chunk 参数
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150