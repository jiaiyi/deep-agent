"""
测试07_1：智能体的整个工作区目录下：只能读文件，不能写（禁止写入）
"""
import os
from deepagents import create_deep_agent, FilesystemPermission
from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

llm = init_chat_model(
    model="deepseek-chat",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)

agent = create_deep_agent(
    model=llm,
    backend=FilesystemBackend(
        root_dir="./backend_dir"
    ),
    permissions=[
        FilesystemPermission(
            operations=["write"],
            paths=["/**"],
            mode="deny"
        )
    ]
)

print("=== 测试：智能体只读权限 ===")

# 测试1：让智能体执行【读取文件】→ 应该允许
result1 = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "读取test.txt文件内容"
    }]
})
print(f"读取结果：{result1['messages'][-1].content}")

# 测试2：让智能体执行【写入文件】→ 应该被拒绝
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "向test.txt写入: hello world"
    }]
})
print(f"写入结果：{result['messages'][-1].content}")
