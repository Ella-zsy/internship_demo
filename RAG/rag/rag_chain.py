from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from retrieval.retriever import get_vector_retriever
from llm.llm_loader import load_llm


retriever = get_vector_retriever()

llm = load_llm()

prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):

    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)