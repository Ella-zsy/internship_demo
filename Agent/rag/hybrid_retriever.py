from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import Chroma
from llm.llm_loader import load_embedding_model
from config import CHROMA_DB_DIR

class HybridRetriever:
    def __init__(self, chunks=None):
        self.vector_db = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=load_embedding_model()
        )
        self.vector_retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})
        
        self.bm25 = None
        self.docs = chunks
        if chunks:
            tokenized = [doc.page_content.split() for doc in chunks]
            self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str):
        # 1. 向量检索
        vector_docs = self.vector_retriever.invoke(query)
        
        # 2. BM25 检索 (如果有解析过的 chunks)
        bm25_docs = []
        if self.bm25 and self.docs:
            bm25_scores = self.bm25.get_scores(query.split())
            top_n = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:2]
            bm25_docs = [self.docs[i] for i in top_n]

        # 3. 去重合并
        all_docs = vector_docs + bm25_docs
        unique_docs = {doc.page_content: doc for doc in all_docs}.values()
        
        return list(unique_docs)