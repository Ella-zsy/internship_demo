from langchain.tools import Tool
from llm.llm_loader import load_llm

llm = load_llm()

def summarize(text):

    prompt = f"Summarize the following text:\n{text}"

    return llm.invoke(prompt)


summarize_tool = Tool(
    name="Summarizer",
    func=summarize,
    description="Summarize long documents"
)