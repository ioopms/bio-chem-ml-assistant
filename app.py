import streamlit as st
import os
import tempfile
import shutil
from datetime import datetime
from study_assistant_dev import StudyAgent, tool_mapping, init_pdf_rag  # 导入 init_pdf_rag

# ---------- 初始化会话状态 ----------
if "agent" not in st.session_state:
    st.session_state.agent = StudyAgent()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "loaded_pdfs" not in st.session_state:
    st.session_state.loaded_pdfs = []          # 存放已加载的PDF路径（显示用）
if "uploaded_pdf_names" not in st.session_state:
    st.session_state.uploaded_pdf_names = set() # 记录已上传的文件名，避免重复

# ---------- 页面配置 ----------
st.set_page_config(page_title="交叉学科学习助手", page_icon="🧬", layout="wide")
st.title("🧪 生物·化学·ML 交叉学科学术助手")
st.caption("基于 DeepSeek + 本地 RAG 知识库 | 支持 PDF 检索、代码执行、笔记归档")

# ---------- 侧边栏 ----------
with st.sidebar:
    st.markdown("### 📚 已加载工具")
    for name in tool_mapping.keys():
        st.markdown(f"- `{name}`")

    st.markdown("---")

    # ---------- PDF 上传区域 ----------
    st.markdown("### 📄 PDF 文献上传")
    uploaded_file = st.file_uploader("选择 PDF 文件", type="pdf", key="pdf_uploader")

    if uploaded_file is not None:
        # 检查是否已经上传过同名文件（避免重复加载）
        file_name = uploaded_file.name
        if file_name not in st.session_state.uploaded_pdf_names:
            # 保存到临时目录（可持久化，但这里用临时文件夹，每次重启会清空）
            # 为了持久，可使用固定目录，但此处演示用 tempfile
            temp_dir = "./temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            # 用时间戳+原文件名避免重名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(temp_dir, f"{timestamp}_{file_name}")

            # 保存文件
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 调用加载函数（会添加到全局向量库）
            with st.spinner(f"正在加载《{file_name}》..."):
                try:
                    # init_pdf_rag 会返回 QA 链，但这里我们只需加载，不需要结果
                    init_pdf_rag(save_path)
                    # 记录已加载
                    st.session_state.loaded_pdfs.append(save_path)
                    st.session_state.uploaded_pdf_names.add(file_name)
                    st.success(f"✅ 《{file_name}》加载成功！")
                except Exception as e:
                    st.error(f"❌ 加载失败：{str(e)}")
                    # 删除可能残留的文件
                    if os.path.exists(save_path):
                        os.remove(save_path)
        else:
            st.info(f"📎 《{file_name}》已加载过，无需重复添加。")

    # ---------- 显示已加载的 PDF 列表 ----------
    if st.session_state.loaded_pdfs:
        st.markdown("### 📑 已加载文献")
        for path in st.session_state.loaded_pdfs:
            # 提取文件名（去掉时间戳前缀）
            display_name = os.path.basename(path).split("_", 1)[-1]  # 取时间戳后的部分
            st.markdown(f"- {display_name}")

    st.markdown("---")

    # ---------- 对话管理 ----------
    st.markdown("### 💾 对话记忆")
    st.markdown(f"当前消息数：{len(st.session_state.messages)}")
    if st.button("清空对话历史"):
        st.session_state.messages = []
        st.session_state.agent = StudyAgent()  # 重置智能体
        st.rerun()

# ---------- 显示历史消息 ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- 接收用户输入 ----------
if prompt := st.chat_input("请输入您的问题（如：帮我总结这篇论文的核心方法）"):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 调用智能体（非流式）
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                # 注入已加载文献列表，让模型获取可用的PDF及完整路径
                loaded_info = "【系统：当前已加载的PDF文献列表】\n"
                if st.session_state.loaded_pdfs:
                    for idx, path in enumerate(st.session_state.loaded_pdfs):
                        display_name = os.path.basename(path).split("_", 1)[-1]
                        loaded_info += f"{idx+1}. 文件名：{display_name}\n   文件路径：{path}\n"
                else:
                    loaded_info += "当前暂无已加载的PDF文献。\n"
                loaded_info += "\n用户问题：" + prompt
                
                response = st.session_state.agent.chat(loaded_info)
            except Exception as e:
                response = f"❌ 发生错误：{str(e)}"
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})