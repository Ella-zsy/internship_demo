from rank_bm25 import BM25Okapi
from retrieval.retriever import get_vector_retriever


class HybridRetriever:

    def __init__(self, documents):

        self.docs = documents

        tokenized = [doc.page_content.split() for doc in documents]

        self.bm25 = BM25Okapi(tokenized)

        self.vector_retriever = get_vector_retriever()

    def search(self, query):

        bm25_scores = self.bm25.get_scores(query.split())

        bm25_top = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:5]

        bm25_docs = [self.docs[i] for i in bm25_top]

        vector_docs = self.vector_retriever.invoke(query)

        return bm25_docs + vector_docs