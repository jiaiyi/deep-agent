"""
旅游规划多智能体（流式输出版）

与 main_agent_graph.py 的区别只在最后一步：用 astream 逐 token 输出。

关键点：
1. stream_mode=["updates","messages"] + subgraphs=True
   - updates  拿到"调用了什么工具/哪个子智能体"
   - messages 拿到 token 级增量
   - subgraphs=True 才能看到子智能体内部（否则只看得到主 Agent）
2. 子智能体靠 meta["lc_agent_name"] 识别，不用解析 namespace 的 uuid
3. 主 Agent 和子智能体的 token 会交错到达，必须分流打印，否则输出会糊成一团
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

AMAP_MCP_URL = "https://mcp.api-inference.modelscope.net/12e53b06dbe748/mcp"
TICKET_MCP_URL = "https://mcp.api-inference.modelscope.net/c885e1a85a914c/mcp"

MAP_TOOL_NAMES = {
    "maps_weather",
    "maps_geo",
    "maps_text_search",
    "maps_around_search",
    "maps_search_detail",
    "maps_direction_walking",
    "maps_direction_driving",
    "maps_direction_transit_integrated",
    "maps_distance",
}

TICKET_TOOL_NAMES = {
    "get-current-date",
    "get-station-code-by-names",
    "get-tickets",
    "get-interline-tickets",
    "get-train-route-stations",
}

llm = init_chat_model(
    model="deepseek-chat",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
)


async def load_mcp_tools(server_name: str, url: str, keep: set[str]):
    client = MultiServerMCPClient(
        {server_name: {"transport": "streamable_http", "url": url}}
    )
    all_tools = await client.get_tools()
    tools = [t for t in all_tools if t.name in keep]
    print(f"[{server_name}] 启用 {len(tools)}/{len(all_tools)} 个工具")
    if not tools:
        raise RuntimeError(f"{server_name} 没有匹配到工具")
    return tools


def build_agent(map_tools, ticket_tools):
    map_agent = {
        "name": "map_agent",
        "description": "查询指定城市/地点的天气、POI 搜索、周边搜索、两地步行/驾车/公交路线与距离时使用。",
        "system_prompt": """
你是高德地图子智能体，只处理地点、路线、周边、距离、POI 详情、天气。

【调用前自检必填参数，缺一个就先追问，绝不空参数调用】
- maps_weather              需要 city
- maps_text_search          需要 keywords（可选 city）
- maps_geo                   需要 address
- maps_around_search        需要 location（必须先由 maps_geo 把地名转成"经度,纬度"）
- maps_search_detail        需要 id
- maps_direction_walking/driving   需要 origin 和 destination（"经度,纬度"格式）
- maps_direction_transit_integrated 需要 origin、destination、city、cityd
- maps_distance             需要 origins 和 destination

【结果校验】
工具返回的城市/地点与请求不符时，视为查询失败，丢弃结果并说明未查询到。
典型反例：maps_geo 空参数会返回广州随机地址。

不得编造地址、距离、耗时、评分、电话、营业时间。
输出只给结论：名称、地址、距离、耗时、路线方案或天气要点，不暴露原始 JSON。
""",
        "tools": map_tools,
    }

    ticket_agent = {
        "name": "ticket_agent",
        "description": "查询火车车次、票价、余票、直达与中转方案、车站名称与电报码、列车经停站时使用。",
        "system_prompt": """
你是 12306 车票子智能体，只负责车次查询、票价分析、直达/中转建议。

【调用前自检必填参数】
- get-current-date            无参数，需要当前日期时先调它
- get-station-code-by-names   需要 stationNames（中文站名）
- get-tickets                 需要 date、fromStation、toStation
- get-interline-tickets       需要 date、fromStation、toStation
- get-train-route-stations    需要 trainCode、departDate

【规则】
1. 用户已给出出发地、目的地、日期中的关键信息时，优先调用工具查询
2. 缺日期先调 get-current-date，按"近期出行"给默认规划
3. 缺出发地不要停止：说明"待补充出发地后可精确查票"，同时给目的地车站与交通预算建议
4. date 格式必须是 YYYY-MM-DD；站名先用 get-station-code-by-names 确认，查不到就追问
5. 预算敏感优先低价，时间敏感优先省时
6. 不做真实购票，只做查询与建议
7. 输出包含：票务状态、推荐方案、预算提示、还需补充的信息
""",
        "tools": ticket_tools,
    }

    summary_agent = {
        "name": "summary_agent",
        "description": "把主 Agent 提供的景点方案与车票方案合并成最终行程。只做整理，不做外部查询。",
        "system_prompt": """
你是旅行方案汇总子智能体。你不做任何外部查询，只整理调用方给你的材料。

规则：
1. 只使用 description 中提供的地图结果和车票结果，不得自行编造或补充未提供的数据
2. 若某一类结果缺失，明确写"暂无该部分数据"，不要猜测
3. 输出结构固定：需求摘要、景点建议、车票建议、预算、行程表、注意事项
4. 给出 1 个推荐方案 + 1 个备选方案
5. 保持简洁，不重复原始数据
""",
        "tools": [],
    }

    return create_deep_agent(
        model=llm,
        system_prompt="""你是一名旅游规划总控智能体。你不能自己查地图或查车票，必须通过 task 工具交给子智能体。

【第一步】从用户输入抽取：出发地、目的地、日期/天数、预算、偏好、出行节奏。
缺失项用默认值：天数=1天、预算=中等、偏好=经典景点+少折腾、节奏=舒适型。
抽取不出的关键信息（尤其是出发地和目的地）先向用户追问。

【第二步】调度子智能体，全部通过 task 工具：
1. 景点、路线、周边、天气 → task(subagent_type="map_agent", description=完整需求)
2. 车票、车次、票价、时间建议 → task(subagent_type="ticket_agent", description=完整需求)

【第三步】汇总。子智能体互相看不到对方，所以你必须把两份结果原文完整拼进 description：
   task(
     subagent_type="summary_agent",
     description="把下面两份结果合并成最终行程：\n【地图结果】<map_agent 返回的全文>\n【车票结果】<ticket_agent 返回的全文>"
   )
不要省略或概括这两段内容。

【第四步】把 summary_agent 的输出整理后：
1. 用 write_file 写入 results/旅游规划-<日期>.md（日期用 YYYY-MM-DD）
2. 同时在回复里输出完整行程

约束：
- 输出必须使用中文
- 不做真实购票，只做规划和建议
- 不得编造地址、距离、耗时、票价
""",
        tools=[],
        subagents=[map_agent, ticket_agent, summary_agent],
        backend=FilesystemBackend(root_dir="travel", virtual_mode=True),
    )


async def stream_run(agent, query: str):
    """流式执行并打印过程"""
    main_buf: list[str] = []
    sub_buf: dict[str, list[str]] = {}
    # 当前正在输出的流，用于判断是否需要打新标签
    current: str | None = None

    async for item in agent.astream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode=["updates", "messages"],
        subgraphs=True,
    ):
        # subgraphs=True 时，chunk 是 (namespace, mode, payload) 三元组
        ns, mode, payload = item if len(item) == 3 else ((), item[0], item[1])

        if mode == "messages":
            msg, meta = payload
            # 只有 AIMessageChunk 才是模型 token 增量；
            # ToolMessage 会带着几万字符的原始 JSON 一起推过来，必须排除
            if type(msg).__name__ != "AIMessageChunk":
                continue

            agent_name = meta.get("lc_agent_name")
            text = msg.content
            # 有的 chunk content 是 list（结构化块），只取文本部分
            if isinstance(text, list):
                text = "".join(
                    b.get("text", "") for b in text if isinstance(b, dict)
                )
            if not text:
                continue

            speaker = agent_name or "main_agent"
            # 主 Agent 和多个子智能体并行时 token 会交错到达，
            # 切换说话人时只打标签，不重打已有内容
            if current != speaker:
                print(f"\n[{speaker}] ", end="", flush=True)
                current = speaker
            if agent_name:
                sub_buf.setdefault(agent_name, []).append(text)
            else:
                main_buf.append(text)
            print(text, end="", flush=True)

        elif mode == "updates":
            for node, upd in (payload or {}).items():
                for msg in ((upd or {}).get("messages") or []):
                    for tc in (getattr(msg, "tool_calls", None) or []):
                        args = tc.get("args", {})
                        if tc["name"] == "task":
                            print(
                                f"\n>>> 调度子智能体 [{args.get('subagent_type')}]"
                                f"  任务: {str(args.get('description', ''))[:60]}..."
                            )
                        else:
                            print(f"\n>>> 调用工具 [{tc['name']}] 参数: {args}")
                # 工具返回
                for msg in ((upd or {}).get("messages") or []):
                    if type(msg).__name__ == "ToolMessage":
                        content = str(getattr(msg, "content", ""))
                        name = getattr(msg, "name", "")
                        if name == "task":
                            print(f"\n<<< 子智能体返回，长度 {len(content)} 字符")
                        elif content:
                            print(f"\n<<< 工具 [{name}] 返回，长度 {len(content)} 字符")

    print(f"\n\n{'#' * 60}")
    print("【子智能体汇总】")
    for name, buf in sub_buf.items():
        text = "".join(buf)
        print(f"\n--- {name}（{len(text)} 字符）---")
        print(text)

    print(f"\n{'#' * 60}")
    print("【最终输出】")
    print("".join(main_buf))


async def main():
    map_tools = await load_mcp_tools("amap-maps", AMAP_MCP_URL, MAP_TOOL_NAMES)
    ticket_tools = await load_mcp_tools("12306", TICKET_MCP_URL, TICKET_TOOL_NAMES)
    agent = build_agent(map_tools, ticket_tools)

    query = await asyncio.to_thread(input, "请输入您的旅游需求: ")
    await stream_run(agent, query)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
