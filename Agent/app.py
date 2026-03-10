import streamlit as st
import os
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from agent.react_agent import get_agent_executor
from rag.dynamic_ingest import process_uploaded_file
from agent.tools import update_retriever
from config import UPLOAD_DIR

st.set_page_config(page_title="Agentic RAG Assistant", page_icon="🤖")

st.title("🤖 Agentic RAG 智能助手")
st.markdown("支持上传文档，自动决定调用检索工具、数学计算或查询时间。")

# 1. 初始化 Session State
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "你好！请在侧边栏上传 PDF，或者直接向我提问。"}]

if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = get_agent_executor()

# 2. 侧边栏：文件上传
with st.sidebar:
    st.header("📄 知识库管理")
    uploaded_file = st.file_uploader("上传 PDF 文档", type=["pdf"])
    
    if uploaded_file:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("正在解析并存入向量库..."):
            chunks = process_uploaded_file(file_path)
            update_retriever(chunks) # 更新混合检索器
        st.success(f"{uploaded_file.name} 入库成功！")

# 3. 渲染历史对话
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 4. 聊天输入与 Agent 处理
# if prompt := st.chat_input("请输入您的问题"):
    
#     # 显示用户输入
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     st.chat_message("user").write(prompt)

#     # 重点：捕捉 Agent 思考过程并展示
#     with st.chat_message("assistant"):
#         st_callback = StreamlitCallbackHandler(st.container())
        
#         try:
#             # 调用 Agent
#             response = st.session_state.agent_executor.invoke(
#                 {"input": prompt},
#                 {"callbacks": [st_callback]}
#             )
#             answer = response["output"]
#         except Exception as e:
#             answer = f"Agent 运行出错：{str(e)}"

#         st.write(answer)
#         st.session_state.messages.append({"role": "assistant", "content": answer})
if prompt := st.chat_input("请输入您的问题..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        
        try:
            response = st.session_state.agent_executor.invoke(
                {"input": prompt},
                {"callbacks": [st_callback]}
            )
            answer = response["output"]
            # 如果小模型把 Prompt 里的词给复读出来了，直接在这里截断！
            cutoff_phrases = [
                "你必须严格遵守", 
                "Question:", 
                "Thought:", 
                "Action:", 
                "【规则】",
                "【标准示例】",
                "---"
            ]
            for phrase in cutoff_phrases:
                if phrase in answer:
                    # 发现复读机的痕迹，立刻切断，只保留前面的有效回答
                    answer = answer.split(phrase)[0].strip()
            
        except Exception as e:
            answer = f"Agent 运行出错：{str(e)}"

        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})