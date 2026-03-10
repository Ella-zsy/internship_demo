from langchain.tools import tool
from datetime import datetime
from rag.hybrid_retriever import HybridRetriever

# 全局变量，用于存放动态更新的 retriever
global_retriever = HybridRetriever()

def update_retriever(chunks):
    """当用户上传新文件时，更新检索器"""
    global global_retriever
    global_retriever = HybridRetriever(chunks)

@tool
def knowledge_base_tool(query: str) -> str:
    """当你被问到与上传的文档、特定知识、具体人物或资料相关的问题时，请使用此工具检索信息。"""
    try:
        docs = global_retriever.search(query)
        if not docs:
            return "知识库中未找到相关信息。"
        context = "\n\n".join([doc.page_content for doc in docs])
        return f"从知识库检索到的参考信息如下：\n{context}"
    except Exception as e:
        return "检索知识库时发生错误。"

@tool(return_direct=True)
def calculator_tool(expression: str) -> str:
    """当你需要进行数学计算时使用此工具。输入必须是有效的数学表达式（例如：'2 + 2 * 3'）。"""
    try:
        # 仅允许基础计算，防止恶意代码
        result = eval(expression, {"__builtins__": None}, {})
        return str(result)
    except Exception as e:
        return "计算错误，请检查表达式是否合法。"

@tool
def current_time_tool(query: str) -> str:
    """当你被问到当前系统时间、今天是几号等时间相关问题时，请使用此工具。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 导出工具列表
agent_tools = [knowledge_base_tool, calculator_tool, current_time_tool]