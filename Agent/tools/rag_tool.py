from langchain.tools import Tool
from vectorstore.chroma_db import load_vectorstore


def rag_search(query):

    vectordb = load_vectorstore()

    docs = vectordb.similarity_search(query, k=3)

    context = ""

    sources = []

    for doc in docs:
        context += doc.page_content + "\n\n"

        if "source" in doc.metadata:
            sources.append(doc.metadata["source"])

    return f"Context:\n{context}\nSources:{sources}"


rag_tool = Tool(
    name="DocumentSearch",
    func=rag_search,
    description="Search information from uploaded documents"
)