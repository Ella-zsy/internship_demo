from fastapi import FastAPI
from rag.rag_chain import rag_chain

app = FastAPI()


@app.get("/")
def root():
    return {"message": "RAG API running"}


@app.post("/chat")
def chat(question: str):

    answer = rag_chain.invoke(question)

    return {
        "question": question,
        "answer": answer
    }