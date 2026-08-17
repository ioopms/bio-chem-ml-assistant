import os
import json
import math
import ast
import traceback
from datetime import datetime
from contextlib import redirect_stdout
import io
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader   # 关键修改
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain.text_splitter import RecursiveCharacterTextSplitter  # 这个仍在 langchain 中
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
#from langchain.chains.retrieval_qa.base import RetrievalQA
#from langchain.chains import RetrievalQA
#from langchain.document_loaders import PyPDFLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_openai import OpenAIEmbeddings, ChatOpenAI
#from langchain_chroma import Chroma
#from langchain.chains import RetrievalQA

load_dotenv()

# ==================== 全局配置 ====================
API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")
MODEL = os.getenv("LLM_MODEL")
# EMBED_MODEL = os.getenv("EMBED_MODEL")
VECTOR_DB_DIR = os.getenv("VECTOR_DB_PATH")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ==================== 全局向量库（持久化 + 增量） ====================
chroma_db = None                     # 全局 Chroma 实例

# ==================== 分离主模型与嵌入模型配置 ====================
# 读取嵌入专用环境变量（如果未设置，则报错提示，防止误用DeepSeek）
EMBED_API_KEY = os.getenv("EMBED_API_KEY")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME")

if not EMBED_API_KEY or not EMBED_BASE_URL:
    raise ValueError("请在 .env 中单独设置 EMBED_API_KEY 和 EMBED_BASE_URL（DeepSeek不支持嵌入，必须用其他付费API）")

# 创建独立的嵌入客户端（走付费API，不占本地显卡）
# 替换原有的 embeddings_model 初始化代码

embeddings_model = OpenAIEmbeddings(
    api_key=EMBED_API_KEY,
    base_url=EMBED_BASE_URL,
    model=EMBED_MODEL_NAME,
    model_kwargs={},           # 确保不传递额外模型参数
    default_headers={},        # 不添加额外请求头
)

def get_or_create_db():
    """获取或创建持久化的全局向量库"""
    global chroma_db
    if chroma_db is None:
        if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
            chroma_db = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings_model)
        else:
            chroma_db = Chroma(embedding_function=embeddings_model, persist_directory=VECTOR_DB_DIR)
            #chroma_db.persist()
    return chroma_db

# ==================== 1. 工具函数 ====================
def calculator(expr: str) -> str:
    """安全数学计算"""
    try:
        allowed = {"sqrt", "pow", "abs", "log", "exp", "pi", "e", "sin", "cos", "tan", "log10"}
        safe_globals = {k: getattr(math, k) for k in allowed if hasattr(math, k)}
        safe_globals["__builtins__"] = None
        res = eval(expr, safe_globals, {})
        return f"【计算结果】{expr} = {res}"
    except Exception as e:
        return f"【计算失败】{str(e)}"

def read_text_file(file_path: str) -> str:
    """读取文本文件（自动尝试多种编码）"""
    for enc in ['utf-8', 'gbk', 'latin-1']:
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            return f"读取失败: {str(e)}"
    return "读取失败: 无法识别文件编码"

def write_md_note(title: str, content: str) -> str:
    """保存 Markdown 笔记（自动按日期归档）"""
    now = datetime.now()
    month_dir = f"study_notes/{now.strftime('%Y-%m')}"
    os.makedirs(month_dir, exist_ok=True)
    filename = f"{now.strftime('%d')}-{title}.md"
    save_path = os.path.join(month_dir, filename)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"笔记已保存至 {save_path}"

# 单个 PDF 查询缓存
rag_cache = {}

def init_pdf_rag(pdf_path: str):
    """加载或增量添加 PDF 至全局向量库，返回对应 QA 链"""
    # 如果已经针对该 PDF 建立过 QA 链，直接返回
    if pdf_path in rag_cache:
        return rag_cache[pdf_path]

    # 加载并拆分文档
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    # 加入到全局数据库
    db = get_or_create_db()
    db.add_documents(split_docs)
    #db.persist()

    # 构建检索器（这里检索仍为全局，如需限定单文档可加 filter，此处保留全局以便交叉参考）
    llm = ChatOpenAI(model=MODEL, api_key=API_KEY, base_url=BASE_URL, temperature=0)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    rag_cache[pdf_path] = qa_chain
    return qa_chain

def rag_query(pdf_path: str, question: str) -> str:
    """基于已加载 PDF 的局部/全局检索（局部仅标记来源，实际检索为全局）"""
    try:
        qa = init_pdf_rag(pdf_path)
        ans = qa.invoke(question)["result"]
        #ans = qa.run(question)
        return f"【PDF检索（来源：{pdf_path}）】\n{ans}"
    except Exception as e:
        return f"RAG查询失败: {str(e)}"

def global_rag_query(question: str) -> str:
    """跨所有已加载文献的全局检索"""
    try:
        db = get_or_create_db()
        llm = ChatOpenAI(model=MODEL, api_key=API_KEY, base_url=BASE_URL, temperature=0)
        retriever = db.as_retriever(search_kwargs={"k": 4})
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        ans = qa.invoke(question)["result"]
        #ans = qa.run(question)
        return f"【全局文献检索】\n{ans}"
    except Exception as e:
        return f"全局检索失败: {str(e)}"

def is_code_safe(code: str) -> tuple[bool, str]:
    """静态检查 Python 代码，禁止危险模块和系统调用"""
    forbidden = {'os', 'subprocess', 'sys', 'shutil', 'importlib', '__import__', 'eval', 'exec'}
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split('.')[0]
                    if root in forbidden:
                        return False, f"禁止导入模块：{alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split('.')[0]
                    if root in forbidden:
                        return False, f"禁止导入模块：{node.module}"
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in forbidden:
                    return False, f"禁止调用函数：{node.func.id}()"
        return True, ""
    except SyntaxError as e:
        return False, f"语法错误：{str(e)}"

def run_safe_python(code: str) -> str:
    """安全执行 Python 代码（支持科学计算、深度学习、图像处理）"""
    safe, err_msg = is_code_safe(code)
    if not safe:
        return f"【代码安全检查不通过】{err_msg}"

    # 构建受限内建函数，保留常用数据操作和 open（用于读写数据文件）
    safe_builtins = {
        'print': print, 'range': range, 'len': len, 'int': int, 'float': float,
        'str': str, 'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
        'bool': bool, 'abs': abs, 'min': min, 'max': max, 'sum': sum,
        'round': round, 'enumerate': enumerate, 'zip': zip, 'map': map,
        'filter': filter, 'sorted': sorted, 'reversed': reversed,
        'isinstance': isinstance, 'type': type, 'Exception': Exception,
        'ValueError': ValueError, 'TypeError': TypeError, 'open': open,
        'True': True, 'False': False, 'None': None,
        'bytes': bytes, 'bytearray': bytearray, 'memoryview': memoryview,
        'complex': complex, 'divmod': divmod, 'pow': pow, 'hash': hash,
        'id': id, 'repr': repr, 'ascii': ascii, 'format': format,
    }

    try:
        capture = io.StringIO()
        local_env = {}
        with redirect_stdout(capture):
            exec(code, {"__builtins__": safe_builtins, "math": math}, local_env)
        output = capture.getvalue()
        return f"【代码输出】\n{output}"
    except Exception:
        err = traceback.format_exc()
        return f"【代码报错】\n{err}"

# 工具映射
tool_mapping = {
    "calculator": calculator,
    "read_text_file": read_text_file,
    "write_md_note": write_md_note,
    "rag_query": rag_query,
    "global_rag_query": global_rag_query,
    "run_safe_python": run_safe_python
}

# ==================== 2. Tool Schema ====================
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "数学表达式计算，如 'sqrt(100) + pi'",
            "parameters": {"type": "object", "required": ["expr"], "properties": {"expr": {"type": "string"}}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_text_file",
            "description": "读取本地 txt/md/csv 文本文件",
            "parameters": {"type": "object", "required": ["file_path"], "properties": {"file_path": {"type": "string"}}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_md_note",
            "description": "保存 Markdown 笔记，自动归档到按日期组织的文件夹中",
            "parameters": {
                "type": "object",
                "required": ["title", "content"],
                "properties": {
                    "title": {"type": "string", "description": "笔记标题（勿带.md后缀）"},
                    "content": {"type": "string", "description": "Markdown 格式内容"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rag_query",
            "description": "基于特定 PDF 文献进行提问（首次使用会自动加载）",
            "parameters": {
                "type": "object",
                "required": ["pdf_path", "question"],
                "properties": {
                    "pdf_path": {"type": "string"},
                    "question": {"type": "string"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "global_rag_query",
            "description": "跨所有已加载文献进行全局知识检索，适合总结、交叉比对",
            "parameters": {
                "type": "object",
                "required": ["question"],
                "properties": {"question": {"type": "string"}}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_safe_python",
            "description": "安全执行 Python 代码，支持 numpy/pandas/matplotlib/seaborn/torch/cv2/sklearn 等，禁止系统操作。用于数据分析、绘图、图像分类、模型训练等。",
            "parameters": {"type": "object", "required": ["code"], "properties": {"code": {"type": "string"}}}
        }
    }
]

# ==================== 3. 智能体主类（含记忆压缩） ====================
class StudyAgent:
    def __init__(self):
        self.system_prompt = """
你是生物·化学·机器学习交叉领域的科研学习智能体。请严格遵守以下规则：
1. 数学计算必须调用 calculator。
2. 阅读论文/课件使用 rag_query（指定PDF路径）或 global_rag_query（跨文献检索）。
3. 数据分析、绘图、图像分类、深度学习实验，调用 run_safe_python 生成并执行代码。
4. 需要保存知识点、错题、文献总结时，调用 write_md_note 自动归档。
5. 读取本地文本文件使用 read_text_file。
6. 输出请使用清晰 Markdown 格式，区分定义、证明、案例、易错点。
7. 引用 PDF 内容时必须注明来源文档，不可编造知识。
8. 每次任务结束后，主动询问是否需要保存笔记。
9. 当用户提到图像分类、目标检测等任务时，可以生成利用 torchvision 或 opencv 的代码，并通过 run_safe_python 执行，前提是用户已提供图像路径。
10. 遇到未知专业术语，优先调用全局文献库检索，或使用可信知识解释。
11. 参考对话开头的【当前已加载的PDF文献列表】获取所有可用的文献完整路径。
12. 当仅加载1篇PDF，且用户提到「这篇论文」「该文献」等指代时，直接使用该PDF路径调用 rag_query，禁止反问用户索要路径。
13. 当加载多篇PDF且用户未指定具体篇目时：
    - 若为跨文献总结、对比分析类需求，优先调用 global_rag_query 进行全局检索；
    - 若为单篇精读需求，再询问用户具体指定哪一篇。
14. 调用 rag_query 时必须使用列表中给出的完整文件路径，不得自行编造路径。
"""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.max_messages_before_compression = 30

    def _compress_history(self):
        """压缩早期对话为上下文摘要，保留最近消息，保证工具调用消息对完整不被截断"""
        if len(self.messages) <= self.max_messages_before_compression:
            return

        system_msg = self.messages[0]
        # 保留最近 8 条消息，增大缓冲减少切断概率
        keep_count = 8
        recent = self.messages[-keep_count:]

        # 关键修复：如果保留的第一条是 tool 消息，说明被截断了，往前多保留1条对应的助手调用消息
        while recent and recent[0]["role"] == "tool":
            keep_count += 1
            recent = self.messages[-keep_count:]

        # 待压缩的历史：系统消息之后、保留部分之前的所有消息
        history = self.messages[1:-keep_count]

        # 构建摘要请求
        summary_prompt = "请将以下学习对话历史压缩为一段简洁的上下文摘要，保留关键概念、结论和未解决的问题：\n"
        for m in history:
            if m['role'] == 'user':
                summary_prompt += f"用户：{m['content']}\n"
            elif m['role'] == 'assistant' and m.get('content'):
                summary_prompt += f"智能体：{m['content']}\n"
        summary_prompt += "\n摘要："

        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.3,
            )
            summary = resp.choices[0].message.content.strip()
        except Exception:
            summary = "（对话摘要生成失败，已保留最近消息）"

        # 重建消息列表
        compressed_system = system_msg["content"] + f"\n\n【历史上下文摘要】{summary}"
        self.messages = [{"role": "system", "content": compressed_system}] + recent

    def chat(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        while True:
            # 每次调用前检查是否需要压缩
            self._compress_history()
            resp = client.chat.completions.create(
                model=MODEL,
                messages=self.messages,
                tools=tools,
                tool_choice="auto"
            )
            msg_obj = resp.choices[0].message
            # 将API返回的对象转为标准字典，保证消息列表格式统一
            # 转为标准字典，自动排除空值字段
            msg_dict = msg_obj.model_dump(exclude_none=True)
            # 双重保险：无工具调用时彻底移除 tool_calls 字段
            if "tool_calls" in msg_dict and not msg_dict["tool_calls"]:
                del msg_dict["tool_calls"]
            
            if not msg_obj.tool_calls:
                self.messages.append(msg_dict)
                return msg_obj.content
            # 处理工具调用
            self.messages.append(msg_dict)
            for call in msg_obj.tool_calls:
                func_name = call.function.name
                args = json.loads(call.function.arguments)
                func = tool_mapping[func_name]
                tool_result = func(**args)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": func_name,
                    "content": tool_result
                })

# ==================== 4. 交互入口 ====================
if __name__ == "__main__":
    import sys
    log_enabled = "--log" in sys.argv
    agent = StudyAgent()
    print("=== 交叉学科学习智能体启动 | 输入 quit 退出 ===")
    if log_enabled:
        log_file = f"chat_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        print(f"对话日志将保存至 {log_file}")

    while True:
        user_in = input("\n你的学习问题：")
        if user_in.strip().lower() == "quit":
            break
        res = agent.chat(user_in)
        print("\n===== 智能体回答 =====")
        print(res)
        if log_enabled:
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(f"用户：{user_in}\n智能体：{res}\n\n")