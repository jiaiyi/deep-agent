"""
测试1：deepagents的第一个测试例子
"""
import os
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from tavily import TavilyClient
from deepagents import create_deep_agent

load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# 工具
@tool
def internet_search(
    query: str,  # 提问
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):

    """使用Tavily进行网络搜索"""
    print("执行internet_search工具")
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

# result = internet_search.invoke({"query": "今天武汉天气怎么样？"})
# print(result)、

# 初始化模型
llm = init_chat_model(
    model="deepseek-chat",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
)

# 创建深度智能体
agent = create_deep_agent(
    model=llm, # 模型
    tools=[internet_search], # 工具
    system_prompt="""你是一位专家级研究员。你的任务是进行深入研究并撰写一份精美的报告。
      你有权使用 internet_search 工具来收集信息。
            """,
)

# 执行agent
result = agent.invoke({"messages": [{"role": "user", "content": "今天武汉天气怎么样?"}]})

# 读取结果数据
print(result)
print(result["messages"][-1].content)