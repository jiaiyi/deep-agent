"""
测试3：异步执行
    agent.ainvoke(): 异步非流式输出
    agent.astream()：异步流式输出
    注意：astream() 返回异步迭代器，必须用 `async for`进行遍历
"""
import asyncio
import os
from typing import Literal

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from openai import api_key
from tavily import TavilyClient

from deep_agent import main

load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# 工具
@tool
def internet_search(
        query: str,
        max_results: int = 5,
        topic:Literal["general","news","finance"] = "general",
        include_raw_content: bool = False,
):
    """使用tavily进行网络搜索"""
    print("执行internet_search工具")
    return tavily_client.search(
        query=query,
        max_results=max_results,
        topic=topic,
        include_raw_content=include_raw_content,
    )

llm = init_chat_model(
    model="deepseek-flash",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)

agent = create_deep_agent(
    model=llm,
    tools=[internet_search],
    system_prompt="""你是一位专家级研究员。你的任务是进行深入研究并撰写一份精美的报告。
      你有权使用 internet_search 工具来收集信息。
            """,
    subagents=[]
)

# 流式调用agent
async def query_astream(query: str):
    async for chunk in agent.astream({"message":[{"role":"user","content":query}]}):
        for key , value in chunk.items():
            if key not in ["model", "tools"]: continue
            # 得到当前message对象
            message = value["messages"][0]
            if key == "model":  # 模型处理的返回
                tool_calls = message.tool_calls
                if tool_calls:  # 准备去调用工具
                    for tool_call in tool_calls:
                        name = tool_call["name"]
                        if name == "task":  # 准备调用子agent (task tool)
                            #  "args": {
                            # #                "subagent_type": "网络搜索助手",  # 目标子 Agent
                            # #                "description": "查询2024政策"     # 下发的具体任务
                            # #              },
                            args = tool_call["args"]
                            print(
                                f"---模型返回，准备调用子智能体[{args['subagent_type']}],传入的参数：{args['description']}")
                        else:  # 准备调用tool
                            #  "name": "read_file_content",
                            # #              "args": {"filename": "需求.docx"},
                            print(f"---模型返回，准备调用工具[{name}], 参数为：{tool_call['args']}")

                else:  # 返回最终的结果
                    print(f"---模型返回最终结果：{message.content}")
            elif key == "tools":  # 工具处理的返回
                name = message.name
                # 子智能体的返回  task tool
                if name == "task":
                    print(f"---子智能执行的返回, 内容: {message.content}")
                # tool的返回
                else:
                    print(f"---工具[name]执行的返回，内容：{message.content}")


# 串行执行
async def test_batch_astream():
    await query_astream("ai最新咨询")
    await query_astream("jev是什么")
    await query_astream("deepagent是什么")

# 并行执行
async def test_batch_astream2():
    cor1 = query_astream("ai最新咨询")
    cor2 = query_astream("jev是什么")
    cor3 = query_astream("deepagent是什么")
    await asyncio.gather(cor1,cor2,cor3)

if __name__ == '__main__':
    asyncio.run(test_batch_astream2())