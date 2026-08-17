# 🧪 生物·化学·ML 交叉学科学术助手
基于 DeepSeek 大模型 + 本地 RAG 知识库的 Streamlit 学术学习助手，支持 PDF 文献检索、安全代码执行、笔记自动归档，面向交叉学科科研学习场景。

## ✨ 功能特性
- **📄 PDF 文献 RAG 检索**：支持多 PDF 增量上传，存入本地 Chroma 向量库，支持单文档精准检索与跨文献全局检索
- **🧠 原生工具调用**：基于 DeepSeek Function Call，自动调度数学计算、文献检索、代码执行、笔记保存等工具
- **🐍 安全 Python 沙箱**：AST 静态语法检测 + 受限执行环境，支持数据分析、绘图、深度学习实验，禁止危险系统调用
- **📝 笔记自动归档**：一键生成 Markdown 笔记，按年月维度自动分类保存到本地
- **💬 对话记忆压缩**：长对话自动摘要压缩，保留核心上下文，有效控制 Token 消耗
- **🖥️ 可视化交互界面**：基于 Streamlit 构建，侧边栏管理文献与工具，对话式交互零成本上手

## 🛠️ 技术栈
- 前端界面：Streamlit
- 大语言模型：DeepSeek V4 (Flash / Pro)
- 嵌入模型：BAAI/bge-m3（SiliconFlow API）
- 向量数据库：Chroma
- RAG 框架：LangChain
- 安全沙箱：AST 静态检测 + 受限内置函数

## 📦 环境要求
- Python 3.10+
- 可正常访问 DeepSeek、SiliconFlow API 的网络环境

## 🚀 快速开始
### 1. 克隆仓库
```bash
git clone https://github.com/ioopms/bio-chem-ml-assistant.git
cd 你的仓库名

pip install streamlit python-dotenv openai langchain langchain-community langchain-openai langchain-chroma langchain-classic pypdf chromadb

# ===== 主模型：DeepSeek =====
LLM_API_KEY=你的DeepSeek API Key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-v4-flash

# ===== 嵌入模型：SiliconFlow =====
EMBED_API_KEY=你的SiliconFlow API Key
EMBED_BASE_URL=https://api.siliconflow.cn/v1
EMBED_MODEL_NAME=BAAI/bge-m3

# ===== 向量库持久化目录 =====
VECTOR_DB_PATH=./chroma_db

streamlit run app.py

.
├── app.py                  # Streamlit 前端主程序，页面交互与会话状态管理
├── study_assistant_dev.py  # 核心智能体逻辑：工具定义、RAG 检索、Agent 调度
├── .env                    # 环境变量与密钥配置（本地私有，不上传仓库）
├── .gitignore              # Git 忽略规则
└── README.md               # 项目说明文档

