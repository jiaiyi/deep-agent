"""
  测试08：使用SKILL
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

# 2. 配置skills
main_agent = create_deep_agent(
    model=llm,
    system_prompt="你是一个智能助手，可以使用SKILL技能！",
    backend=FilesystemBackend(root_dir="./"),
    skills=["./skills"]
)

# 执行
result = main_agent.invoke(
    {
        "messages": [
            # {"role": "user", "content": "列出我的所有技能"}
            # {"role": "user", "content": "将‘我今天要坐飞机去北京潇洒一下’翻译为表情包"},
            {"role": "user", "content": "审查一下我当前项目下08_skill.py的代码"}
        ]
    },
)

print(f"最终输出结果：{result['messages'][-1].content}")