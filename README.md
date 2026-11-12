# Deep Agent · 从能力清单到可用的多智能体应用

> 一个 `deepagents`（LangGraph）的实战仓库：
> 前面 20 个编号脚本，一个能力一个可跑的最小例子；最后一个 `travel/` 把能力拼成能用的东西 ——
> 一个查地图、查火车票、最后出规划书的旅行助手。

约 **2,000 行 Python**。适合想搞清楚「Agent 的权限、子智能体、人工审批到底怎么落地」的人直接跑。

---

## 1. 为什么不直接看文档

文档告诉你有哪些参数，不告诉你**什么时候它会不按你想的工作**。这个仓库按能力分块，
每块都是能单独运行、能改参数看结果的最小例子：

| 编号 | 主题 | 看点 |
|---|---|---|
| `01` | Hello World | 最小的 `create_deep_agent` |
| `02` / `03` | 流式 / 异步流式输出 | 增量拿 token，而不是等整段返回 |
| `04-1` ~ `04-5` | 子智能体 | 基本用法、兼容 LangChain、兼容 LangGraph、结构化输出、**嵌套子智能体** |
| `05-1` / `05-2` | 人工审批 | 高危工具挂起等人批；审批时**修改参数**再执行 |
| `06-1` ~ `06-3` | Backend | 纯文件、Store、两者组合 —— Agent 的文件到底存在哪 |
| `07-1` / `07-2` | 权限 | 全局只读、目录隔离 |
| `08` | Skills | 把可复用流程写成 `SKILL.md`，按需加载 |
| `09-1` / `09-2` | Context | 输入上下文与运行时上下文 |

`skills/` 下有两个可直接用的示例技能：`code-reviewer`、`emoji-translator`。

---

## 2. 两个容易被忽略的点

**权限要在代码里，不在提示词里。** 在 system prompt 写「不要删除文件」只是建议，
模型完全可以把它理解成「这次用户明确要求了，那就删」。真正管用的是
`FilesystemPermission` 这种代码级约束 —— 模型看不到规则也得遵守：

```python
# 07_permission2_目录隔离.py
permissions=[
    FilesystemPermission(operations=["read", "write"], paths=["/allow_dir/**"], mode="allow"),
]
```

**审批不只是「同意 / 拒绝」。** `05_人工审批2_修改.py` 演示的是：工具调用被挂起后，
人可以改参数再放行 —— 比如模型要删 `user` 表，人不批原样，而是把参数改成 `user2` 再执行
（也演示了只允许 `approve|reject`、不允许改参数的工具该怎么配）。这比一刀切的拒绝实用得多。

---

## 3. 综合案例：旅行规划助手

`travel/` 把上面的能力拼成一个真实应用：主 Agent 带三个子 Agent，
其中两个接的是**远端 MCP Server**。

```
main_agent
  ├─ task → map_agent     ──MCP──▶ 高德地图（天气/POI/周边/路线/距离）
  ├─ task → ticket_agent  ──MCP──▶ 12306（车次/票价/余票/中转）
  └─ task → summary_agent         把前两份结果整理成行程与预算
                       └──▶ results/旅游规划-*.md
```

几个实现上的取舍：

- **工具白名单**。MCP 一次返回几十个工具，全挂上去模型会挑错、还会乱填参数；
  地图只留 9 个（天气、地理编码、关键字/周边搜索、详情、步行/驾车/公交路线、距离），
  12306 只留查票相关的几个。
- **子 Agent 的 description 写得像「转诊说明」**。不说「地图助手」，
  而说「任何需要真实地理位置、地址、坐标、距离、耗时或天气的问题都交给它」——
  主 Agent 靠这段话决定要不要转派。
- **禁止编造**。地图和票务两个子 Agent 的提示词里都明确写了：
  查不到就说查不到，不许编地址、距离、票价、营业时间。
- **产物落盘**。最终规划写成 `results/旅游规划-<日期>.md`，会话结束还能回看。

跑之前需要 `DASHSCOPE_API_KEY`（模型走百炼兼容端点）。MCP 地址填的是魔搭的托管实例，
换成自己的也可以，改文件顶部的两个常量即可。

---

## 4. 快速开始

```bash
# Python 3.11+
uv sync

cp .env.example .env       # 填入 DASHSCOPE_API_KEY / BASE_URL，以及可选的 TAVILY、LangSmith
python 01_helloworld.py    # 从最小的开始，按编号往下跑
python travel/main_agent_graph.py
```

`.env`、`.venv`、`backend_dir/` 都已在 `.gitignore` 里 —— **不要把 key 提交上去**。

---

## 5. 目录

```
01~09_*.py            按能力编号的最小示例
travel/               旅行规划多智能体（主 Agent + 3 个子 Agent）
skills/               示例技能：code-reviewer / emoji-translator
backend_dir/          Agent 的可写目录（目录隔离示例用）
src/deep_agent/       包入口
```

---

## 6. 相关文章

- [给 Agent 划权限边界：全局只读、目录隔离、人工审批](https://jiaiyi.github.io/posts/agent-permission-boundaries.html) ·
  [掘金版](https://juejin.cn/post/7706822082815623202)
- 更多项目：[作品集](https://jiaiyi.github.io/portfolio.html) · [博客](https://jiaiyi.github.io)
