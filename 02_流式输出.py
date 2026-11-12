"""
测试2：流式输出
"""
import os
from typing import Literal

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from tavily import TavilyClient

load_dotenv()
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# 工具
@tool
def internet_search(
        query:str,
        max_results:int = 5,
        topic:Literal["general","news","finance"] = "general", #网页搜索走什么类型库
        include_raw_content:bool = False, # 返回结果带不带网页正文全文
):

    """使用Tavily进行网络搜索"""
    print("执行internet_search工具")
    return tavily_client.search(
        query=query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

# 初始换模型

llm = init_chat_model(
    model="deepseek-flash",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)

# 创建深度智能体
agent = create_deep_agent(
    model=llm,
    tools=[internet_search],
    system_prompt="""
    你是一位专家级研究员。你的任务是进行深入研究并撰写一份精美的报告。
      你有权使用 internet_search 工具来收集信息。"""
)


# 流式执行agent
def query_stream(query:str):
    for chunk in agent.stream({"messages":[{"role":"user","content":query}]}):
        for key , value in chunk.items():
            if key not in ["model","tools"]: continue
            # 得到当前的Message对象
            message = value["messages"][0]
            if key == "model": #处理模型返回
                tool_calls = message.tool_calls
                if tool_calls: #准备调用工具
                    for tool_call in tool_calls:
                        name = tool_call["name"]
                        if name == "task": #调用子agent
                            args = tool_call["args"]
                            print(f"---模型返回，准备调用子智能体[{args['subagent_type']}],传入的参数：{args['description']}")
                        else: #调用tool
                            print(f"---模型返回，准备调用工具[{name}], 参数为：{tool_call['args']}")
                else: #模型返回结果
                    print(f"----模型返回最终结果:{message.content}")
            elif key == "tools":#处理工具调用返回
                name = message.name
                if name == "task": #子agent的返回
                    print(f"---子智能执行的返回, 内容: {message.content}")
                else: #tool的返回
                    print(f"---工具执行的返回，内容：{message.content}")


if __name__ == '__main__':
    query_stream("今天武汉天气怎么样")