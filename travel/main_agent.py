import asyncio
import os

from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

llm = init_chat_model(
    model="deepseek-chat",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)


async def build_map_agent():
    """连接魔搭高德 MCP，把远端工具转成 langchain 工具"""
    map_client = MultiServerMCPClient(
        {
            "amap-maps": {
                "transport": "streamable_http",
                "url": "https://mcp.api-inference.modelscope.net/12e53b06dbe748/mcp",
            }
        }
    )
    map_tools = await map_client.get_tools()
    print("已加载地图工具:", [map_tool.name for map_tool in map_tools])



    map_agent = {
    "name": "map_agent",
    "description":"查询天气、搜索景点/餐厅/酒店等地点、查周边设施、查两地之间的步行/驾车/公交路线与距离、查地点详情时使用。任何需要真实地理位置、地址、坐标、距离、耗时或天气的问题都交给它。",
    "system_prompt":"""
                    你是高德地图 MCP 助手。
                    只回地点、路线、周边、距离、POI 详情和天气问题。
                    
                    优先使用高德地图 MCP 工具；缺少城市、起点、终点、关键词或范围时先追问。
                    不得编造地址、距离、耗时、评分、电话或营业时间。
                    工具返回为空或不完整时，说明未查询到相关信息，并提示需要补充的内容。
                    非地图类问题不调用地图工具。
                    
                    输出时只给结论：名称、地址、距离、耗时、路线方案或天气要点；不暴露原始 JSON。
                    """,
    "tools": map_tools,
    }

    return map_agent

async def build_ticket_agent():
    ticket_client = MultiServerMCPClient(
        {
            "12306": {
                "transport": "streamable_http",
                "url": "https://mcp.api-inference.modelscope.net/c885e1a85a914c/mcp"
            }
        }
    )
    ticket_tools = await ticket_client.get_tools()
    print("已加载查票工具",[ticket_tool.name for ticket_tool in ticket_tools])

    ticket_agent = {
        "name": "ticket_agent",
        "description":"查询火车车次、票价、余票、运行时长、直达或中转方案、车站名称与电报码、列车经停站时使用。任何涉及坐火车、高铁、查车次或算交通预算的问题都交给它。",
        "system_prompt":"""
                        你是一名12306车票规划助手。
                        你只负责车次查询、票价分析、直达/中转建议。
                        
                        规则：
                        1. 只要用户已经给出出发地、目的地、出行日期中的关键信息，就优先调用12306相关工具查询
                        2. 如果缺少出行日期，先调用当前日期工具，再按“近期出行”给出默认规划
                        3. 如果缺少出发地，不要直接停止；先给出“待补充出发地后可精确查票”的说明，同时尽量补充目的地车站和交通预算建议
                        4. 如预算敏感，优先给出低价方案；如时间敏感，优先给出省时方案
                        5. 不做真实购票，只做查询与建议
                        6. 输出必须包含：票务状态、推荐方案、预算提示、还需补充的信息
                        """,
        "tools": ticket_tools,
    }

    return ticket_agent

summary_agent = {
    "name": "summary_agent",
    "description":"map_agent 和 ticket_agent 都完成后，把主智能体传入的景点方案与车票方案合并成最终行程时使用。它不做任何外部查询，只整理已有材料。",
    "system_prompt":"""
                    你是一名旅行方案汇总助手。
                    你不负责外部查询，只负责整理结果。
                    
                    规则：
                    1. 将景点方案与车票方案合并
                    2. 输出最终推荐方案、备选方案、预算估算
                    3. 按“需求摘要、景点建议、车票建议、预算、行程表、注意事项”结构输出
                    4. 保持简洁，不重复原始数据
                    """
}


map_agent = asyncio.run(build_map_agent())
ticket_agent = asyncio.run(build_ticket_agent())


main_agent = create_deep_agent(
    model=llm,
    system_prompt="""你是一名旅游规划总控智能体。
                    你的职责是根据用户输入，先抽取关键信息，再调度合适的子智能体完成任务。
                    注意: 必须将规划好的内容写到 results文件夹/旅游规划-日期.md文件
                    规则：
                    1. 先从用户输入中抽取：出发地、目的地、日期/天数、预算、偏好、出行节奏
                    2. 如果用户没有明确说明游玩天数，默认按 1 天规划
                    3. 如果用户没有明确说明预算，默认按中等预算规划
                    4. 如果用户没有明确说明偏好，默认按“经典景点 + 少折腾”规划
                    5. 如果用户没有明确说明出行节奏，默认按“舒适型节奏”规划
    
                    6. 景点、路线、地图相关问题交给 map_agent
                    7. 火车票、车次、票价、时间建议交给 ticket_agent
                    8. map_agent 和 ticket_agent 完成后，调用 summary_agent 汇总。注意子智能体之间互相看不到，
                       你必须把这两份结果原文完整拼进 task 的 description 里，不要省略或概括：
                       task(subagent_type="summary_agent",
                            description="把下面两份结果合并成最终行程：\n【地图结果】<map_agent 返回的全文>\n【车票结果】<ticket_agent 返回的全文>")
                    9. 输出必须使用中文
                    10. 不做真实购票，只做规划和建议
                """,
    backend=FilesystemBackend("./"),
    memory=["AGENTS.md"],
    subagents=[map_agent,ticket_agent, summary_agent]
)



async def run_main_agent():
    query = input("请输入你的需求：").strip()
    print("\n=================开始规划=================")
    final_content = ""
    subagent_names_dict:dict[str,str] = {}
    async for chunk in main_agent.astream({"messages":[{"role":"user","content":query}]}):
        for key,value in chunk.items():
            if key not in ["model","tools"]:continue
            message = value["messages"][0]
            if key == "model":
                tool_calls = message.tool_calls
                if tool_calls:
                    for tool_call in tool_calls:
                        name = tool_call["name"]
                        if name == "task":
                            args = tool_call["args"]
                            print(f"\n---模型返回，准备调用子智能体[{args['subagent_type']}],传入的参数：{args['description']}")
                            subagent_names_dict[tool_call["id"]] = args["subagent_type"]
                        else:
                            print(f"\n---模型返回，准备调用工具[{name}], 参数为：{tool_call['args']}")
                else:
                    final_content = message.content
            elif key == "tools":
                name = message.name
                if name == "task":
                    subagent_name = subagent_names_dict[message.tool_call_id]
                    print(f"\n---子智能[{subagent_name}]执行的返回, 内容: {message.content[:100]}")
                else:
                    print(f"\n---工具[{name}]执行的返回，内容：{message.content[:100]}")
    print("\n=======最终结果=======")
    print(final_content or "本次没有生成最终结果")

if __name__ == "__main__":
    asyncio.run(run_main_agent())