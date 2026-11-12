"""
测试4_1: 子智能体的基本使用  => 基于字典定义子agent
创建一个主智能体，它拥有三个助手：
    1.  **天气助手**：查询天气（固定返回“晴朗”）。
    2.  **计算助手**：处理数学问题。
    3.  **翻译助手**：负责中英互译。
"""

import os
from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# 加载.env配置
load_dotenv()

# 初始化模型
llm=init_chat_model(
    model="deepseek-chat",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
)



weather_llm = init_chat_model(
    model="deepseek-chat",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
)

weather_agent = {
    "name":"weather_helper",
    "description":"你是一个天气查询助手",
    "system_prompt":"你是一个天气助手, 直接返回天气信息:'今天天气晴，温度：20度'",
    "model" : weather_llm,
}

math_agent = {
    "name":"math_helper",
    "description":"一个用于处理数学计算的助手",
    "system_prompt":"你是一个严谨的数学助手，回答用户所有的数学计算问题",
    "tools":[],
}

translate_agent = {
    "name":"translate_helper",
    "description":"一个中英文翻译助手",
    "system_prompt":"你是一个中英想到翻译助手，如果是中文翻译为英文，反之翻译为中文",
    "tools":[],
}

# 创建深度代理
agent = create_deep_agent(
    model = llm,
    system_prompt="你是一个强大的助手",
    tools=[],
    subagents=[weather_agent, math_agent, translate_agent],
)
def query_stream(query: str):
    for chunk in agent.stream({
        "messages": [{"role": "user", "content": query}]
    }):
        # print(chunk)
        for key, value in chunk.items():
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


if __name__ == '__main__':
    query_stream("今天武汉天气怎么样? 再计算123+234-99等于几？，同时将'好好学习，天天向上'翻译为英文")
