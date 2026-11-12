"""
  测试4_3: 兼容langgraph子智能体格式
"""
import os
from typing import TypedDict, Annotated, Literal

from dotenv import load_dotenv
from deepagents import create_deep_agent, CompiledSubAgent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langgraph.constants import END, START
from langgraph.graph import add_messages, StateGraph
from tavily import TavilyClient

load_dotenv()

chat_model = init_chat_model(
    model="deepseek-chat",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """使用 Tavily 进行联网搜索"""
    print(f"正在搜索,query={query}")
    return tavily_client.search(
        query=query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

# 创建langgraph的状态图
class MyState(TypedDict):
    # 注解messages的类型为list, messaages的数据在多个节点之间是累加的方式
    messages: Annotated[list, add_messages]
def process_node(state: MyState):
    print(f"--开始处理节点：{state}")
    query = state['messages'][-1].content
    result = internet_search(query)

    return {"messages": [AIMessage(f"经过节点处理后的结果：原数据为：{result}")]}

state_graph = StateGraph(state_schema=MyState)
state_graph.add_node("process_node", process_node)
state_graph.add_edge(START, "process_node")
state_graph.add_edge("process_node", END)
compiled_graph = state_graph.compile()

# 将langchain的智能体转为深度代理的子智能体
custom_subagent = CompiledSubAgent(
    name="weather_subagent",
    description="你是一个天气查询助手的子智能体,自己返回天气信息",
    runnable=compiled_graph
)
# 创建主智能体，设置子智能体
main_agent = create_deep_agent(
    model=chat_model,
    subagents=[custom_subagent],
    system_prompt="你是一个智能助手，所有的业务动作，你不要亲自做，都找weather_subagent子智能体进行处理！"
)

print("开始执行任务...")
result = main_agent.invoke({
    "messages": [
        {"role": "user", "content": "武汉今天的天气怎么样？"}
    ]
})
print(f"最终结果：{result['messages'][-1].content}")
