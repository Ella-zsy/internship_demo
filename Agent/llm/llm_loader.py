from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings
from config import LLM_MODEL_PATH, EMBEDDING_MODEL

def load_llm():
    """加载本地 llama.cpp 模型"""
    llm = LlamaCpp(
        model_path=LLM_MODEL_PATH,
        temperature=0.1,      
        n_gpu_layers=100,
        n_batch=512,
        n_ctx=4096,           
        f16_kv=True,
        verbose=False
    )
    return llm

def load_embedding_model():
    """加载 HF Embedding 模型"""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )