"""
测试09_1：理解各个输入上下文
"""
import os
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model(
    model="deepseek-chat",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)

agent = create_deep_agent(
    model=llm,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "帮我写一份中秋节节日祝福!"}]}
)
print(f"[Agent]: {result['messages'][-1].content}")
