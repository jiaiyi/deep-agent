"""
  测试06_1: 演示使用fileSystemBackEnd实现长期记忆
   实现跨会话的数据共享
"""
import os

from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from deepagents.backends import FilesystemBackend

load_dotenv()

llm = init_chat_model(
    model="deepseek-chat",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)


main_agent = create_deep_agent(
    model=llm,
    tools=[],
    system_prompt="""
        你是一个智能助手，可以使用文件工具进行文件读写操作！但是只有在用户明确要求的情况下，你才可以创建文件
    """,
    backend=FilesystemBackend(root_dir="./backend_dir",virtual_mode=True)
)

result = main_agent.invoke({
    "messages": [
        {"role": "user", "content": "帮我查询下python语言的介绍,并且帮我写到 xxx.txt文件"}
    ]
})
print(f"最终结果: {result['messages'][-1].content}")

result2 = main_agent.invoke({
    "messages": [
        {"role": "user", "content": "堆区xxx.txt文件"}
    ]
})
print(f"最终结果: {result2['messages'][-1].content}")
