import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # 加载 .env

client = OpenAI(
    api_key=os.getenv("EMBED_API_KEY"),
    base_url=os.getenv("EMBED_BASE_URL")
)

try:
    response = client.embeddings.create(
        model=os.getenv("EMBED_MODEL_NAME"),
        input="你好，测试一下",
    )
    print("✅ API 调用成功！")
    print("向量维度:", len(response.data[0].embedding))
except Exception as e:
    print("❌ 调用失败：", e)