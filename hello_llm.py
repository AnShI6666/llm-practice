import os
from openai import OpenAI

# 1. 从环境变量读取 API Key
api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("请先设置 DEEPSEEK_API_KEY 环境变量")

# 2. 初始化 DeepSeek 客户端（兼容 OpenAI SDK）
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"  # DeepSeek 的接口地址
)

# 3. 发送请求
response = client.chat.completions.create(
    model="deepseek-chat",  # 使用 DeepSeek-V3 模型
    messages=[
        {"role": "user", "content": "用一句话解释什么是大语言模型。"}
    ],
    stream=False
)

# 4. 打印结果
print(response.choices[0].message.content)