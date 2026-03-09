from vectorstore.chroma_db import load_vectorstore


def get_vector_retriever():

    vectordb = load_vectorstore()

    retriever = vectordb.as_retriever(
        search_kwargs={"k": 5}
    )

    return retriever