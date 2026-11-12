"""
  测试4_2: 兼容langchain子智能体格式
"""
import os
from typing import Literal

from dotenv import load_dotenv
from deepagents import create_deep_agent, CompiledSubAgent
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from tavily import TavilyClient

load_dotenv()

chat_model = init_chat_model(
    model="deepseek-chat",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# 工具调用
def get_weather(
        city:str,
        max_results:int=10,
        topic:Literal["general","news","finance"] = "general",
        include_raw_content:bool=False
):
    """查询指定城市天气"""
    print(f"get_weather工具正在查询{city}天气")
    return tavily_client.search(
        query=city,
        max_results=max_results,
        topic=topic,
        include_raw_content=include_raw_content
    )

chain_agent = create_agent(
    model=chat_model,
    system_prompt="你是一个查询天气的助手,调用工具获取天气数据",
    tools=[get_weather]
)
# 讲langchain智能体保证为deepagents支持的子智能体
sub_agent = CompiledSubAgent(
    name="sub_agent",
    description="你是一个查询天气的子助手",
    runnable=chain_agent
)


# 创建主智能体，设置子智能体
main_agent = create_deep_agent(
    model=chat_model,
    subagents=[sub_agent],
    system_prompt="你是一个智能助手，所有的业务动作，你不要亲自做，都找子智能体进行处理！"
)

print("====开始执行任务====")
result = main_agent.invoke({
    "messages": [
        {"role": "user", "content": "武汉今天的天气怎么样？"}
    ]
})
print(f"最终结果：{result['messages'][-1].content}")