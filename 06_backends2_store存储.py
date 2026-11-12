"""
  测试06_2: 演示使用StoreBackend实现长期记忆
  实现跨会话的数据共享
"""
from deepagents import create_deep_agent
from deepagents.backends import StoreBackend, StateBackend, FilesystemBackend, CompositeBackend, store
from langgraph.store.memory import InMemoryStore
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model
import os
load_dotenv()

llm = init_chat_model(
    model="deepseek-chat",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)

store = InMemoryStore()
main_agent = create_deep_agent(
    model=llm,
    system_prompt="""
    你要把用户的重要信息保存到user_profile.txt文件中！
    获取用户信息可以读取user_profile.txt文件！
    """,
    store=store,
    backend=StoreBackend(
        namespace=lambda rt:("ns_name",)
    ),
)


config_a  = {"configurable":{"thread_id":"a"}}
config_b  = {"configurable":{"thread_id":"b"}}

#第一遍执行 =>保存一些重要数据
result_a = main_agent.invoke(
    {
        "messages":[{"role":"user","content":"我是大风哥，我今年19岁！"}]
    },
    config=config_a
)
print(f"第1次回复结果：{result_a['messages'][-1].content}")


#第二遍执行 -》 读取一些重要信息
result_b = main_agent.invoke(
    {
        "messages":[{"role":"user","content":"我叫啥和我的年龄！"}]
    },
    config=config_b
)
print(f"第2次回复结果：{result_b['messages'][-1].content}")


# 手动读取store存储的数据
result = store.search(("ns_name",))
print("-----",result)
for item in result:
   print("===",item.namespace,item.value,item.key)