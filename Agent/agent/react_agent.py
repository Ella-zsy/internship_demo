from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from llm.llm_loader import load_llm
from agent.tools import agent_tools

QWEN_REACT_PROMPT = """你是一个聪明的 AI 助手。请尽力解答用户的问题。你可以使用以下工具：

{tools}

你必须严格遵守以下思考和行动的格式：

Question: 用户提出的问题
Thought: 思考我接下来需要做什么
Action: 工具名称，必须是 [{tool_names}] 之一
Action Input: 传入工具的参数（仅填参数值，不要加引号或参数名！）
Observation: 工具返回的结果
Thought: 我现在知道最终答案了
Final Answer: 直接给出对用户问题的最终回答

---
【标准示例】
Question: 15乘以85等于几？
Thought: 我需要计算15和85的乘积。
Action: calculator_tool
Action Input: 15*85
Observation: 1275
Thought: 我现在知道最终答案了。
Final Answer: 15乘以85等于1275。
---

【规则】：
1. 每次只准采取一个 Action，不要自己编造 Observation。
2. 只要你输出了 `Action Input:` 的值，必须立刻停止生成。
3. 当你得到 Observation 并得出结论后，必须使用英文字母 `Final Answer:` 作为最终回答的开头！绝对不允许写成“最终答案：”、“回答：”或直接输出内容。
4. 一旦输出了 `Final Answer:` 及其内容，必须立刻停止生成。

开始！

Question: {input}
Thought: {agent_scratchpad}"""

def get_agent_executor():
    llm = load_llm()
    tools = agent_tools
    
    prompt = PromptTemplate.from_template(QWEN_REACT_PROMPT)
    
    # 修改错误提示，让它知道下一步该干嘛
    def handle_parsing_error(error: Exception) -> str:
        return (
            "Observation: 格式错误！请检查你的输出。\n"
            "如果你需要调用工具，请只输出 Action 和 Action Input。\n"
            "如果你已经知道答案，请只输出 Final Answer。"
        )

    agent = create_react_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,          
        handle_parsing_errors=handle_parsing_error,
        max_iterations=3
    )
    
    return agent_executor