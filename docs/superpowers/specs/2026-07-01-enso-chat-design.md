# ENSO 对话式 Agent 设计

日期：2026-07-01

## 1. 背景与目标

现有 `D:\作品\agent\` 是一次性 fire-and-forget 的 ENSO 预测 agent：跑完整流水线出报告，用户不能中途插话。
本设计在 `D:\作品\` 下新建**独立项目 `enso-chat/`**，做一个**全对话式 ENSO agent**：用户在 Streamlit chat 界面自由对话，agent 用 DeepSeek 驱动现有 ENSO 工具（预测 / 诊断 / 画图 / 分析），每轮返回控制权给用户，用户随时插话改方向。

定位：课程/原型级对话式 agent。不是业务系统。

## 2. 设计决策（已与用户确认）

| 决策点 | 选择 | 理由 |
|---|---|---|
| 项目位置 | `D:\作品\enso-chat/` 独立平级 | 与 `agent/` 平级，互不影响 |
| 复用方式 | 完全独立复制 `agent/src/`，可删减重构 | 新项目自带一份 src/，不动原 `agent/` |
| 对话入口 | Streamlit chat（`st.chat_input` + `st.chat_message`） | 原生支持对话，能嵌图表，用户会用 |
| loop 路线 | 流式 turn-by-turn（不做 `ask_user` 暂停） | 用户主导对话，改动中等，贴合现有结构 |
| 会话持久化 | 不持久化（仅 session_state 内存） | 原型够用，关页面即丢 |
| 工具展示 | 折叠展示工具步骤（expander 显示调用+结果） | 界面干净又透明 |
| 工具集 | 精简对话工具集（去 4 个重流水线工具） | 对话场景更轻 |
| 历史超限 | 自动摘要压缩 | 多轮后保持上下文 |
| loop 实现 | 新写 `run_turn`，不碰现有 `run_agent` | 职责单一，零回归风险 |
| offline 模式 | 删 OfflineClient，只留 DeepSeekClient | 对话本质需 LLM；无 key 时禁用对话输入 |

## 3. 架构与数据流

```
[Streamlit chat]
  用户输入 ──► st.chat_input
      │
      ▼
  session_state.messages = [system, ...历史, 刚加的user]
      │
      ├─ 历史超 token? ──► summarize_old_messages() ──► 压缩旧轮为摘要
      │
      ▼
  run_turn(messages, tools, client, on_step=渲染回调)
      │  while step < max_steps:
      │    调 DeepSeek (复用 _chat_with_retry)
      │    assistant 回复 (可能带工具调用)
      │    if 不调工具 → return (messages, 文字)   ◄── 本轮结束,控制权回用户
      │    执行工具 (复用 _execute_tool_calls)
      │    on_step 回调往气泡流插 expander
      │    结果喂回 messages
      ▼
  渲染 assistant 文字气泡
  等用户下一条 ──► 回到顶部
```

关键边界：

- 状态 = `session_state.messages`（纯内存，关页面即丢）。
- 每轮 `run_turn` 跑到"模型不调工具"就返回，把控制权交回用户。
- 工具调用过程通过 `on_step` 回调实时插折叠块，结果也进折叠块。
- DeepSeek 默认必需（offline 无法对话，无 key 时禁用对话输入）。

## 4. 项目结构与工具集

### 4.1 目录结构

```text
D:\作品\enso-chat\
  README.md
  requirements.txt          (streamlit + 现有科学栈)
  pyproject.toml            (pytest 配置,沿用 agent 风格)
  src\
    __init__.py
    config.py               (路径 + DeepSeek 配置,精简自 agent)
    data\                   (复制:sample_generator, noaa_enso, loaders)
    features\               (复制:enso_features)
    analysis\               (复制:enso_phase, precipitation_analysis)
    models\                 (复制:baseline, enso_ml, evaluation, tide_model)
    visualization\          (复制:plots)
    agent\
      __init__.py
      client.py             (复制:DeepSeekClient + LLMClient 协议,原样)
      tools.py              (复制 + 精简:去 write_report/build_enso_features,
                              保留 forecast_for_month/diagnose/recommend_dict/
                              classify_phase/read_results/画图类/降水/潮汐)
      run_turn.py           (★ 新写:turn-by-turn loop)
      summarizer.py         (★ 新写:历史超限摘要压缩)
    web\
      __init__.py
      app.py                (★ 新写:Streamlit chat 界面)
      chat_helpers.py       (★ 新写:纯函数,渲染/解析/消息构造,可测)
  tests\
    test_run_turn.py        (★ 新写)
    test_summarizer.py      (★ 新写)
    test_chat_helpers.py    (★ 新写)
    test_tools.py           (复制 + 精简:只测保留的工具)
```

### 4.2 精简后的工具集（对话高频）

| 保留 | 去掉 |
|---|---|
| `load_enso_data`（数据加载，基础） | `write_report`（重流水线，对话场景不写报告） |
| `forecast_for_month`（目标月预测，核心） | `build_enso_features`（load_enso_data 已含） |
| `diagnose_local_data`（数据诊断） | `train_and_evaluate`（forecast_for_month 已含评估） |
| `recommend_data_range`（数据范围建议） | （另：`OfflineClient` 整个删掉） |
| `forecast_latest`（1/3/6 月预测） | |
| `classify_phase`（阶段分类） | |
| `read_results`（读结果） | |
| 画图 4 个（timeseries/obs_vs_pred/rmse/phase） | |
| `analyze_precipitation`、`run_tide_prediction` | |

保留约 12 个，去掉 4 个重流水线工具。报告生成（`reporting/` 整个目录）对话场景不需要，**不复制**。

### 4.3 复制方式

`agent/src/` 下 `data/features/analysis/models/visualization/agent/client.py` + `agent/tools.py` 原样复制，然后从 `tools.py` 的 `build_tools` 删掉 4 个工具及其实现函数。`run_turn.py`、`summarizer.py`、`web/` 全新写。`config.py` 精简自 agent（删 reporting 路径常量、OfflineClient 相关）。

## 5. `run_turn` —— turn-by-turn loop

### 5.1 接口

```python
def run_turn(
    messages: list[dict],          # 已含 system + 历史 + 刚追加的 user 消息
    tools: ToolRegistry,
    client: LLMClient,
    *,
    on_step: Callable[[int, str, dict, str], None] | None = None,
    # (step, tool_name, arguments, result) 每个工具调用触发,供 UI 渲染折叠块
    max_steps: int = AGENT_MAX_STURNS,      # 单轮内最多调几次工具(防失控)
    loop_limit: int = AGENT_LOOP_LIMIT,    # 循环检测
    max_retries: int = AGENT_MAX_RETRIES,
    base_delay: float = AGENT_RETRY_BASE_DELAY,
    max_delay: float = AGENT_RETRY_MAX_DELAY,
) -> TurnResult:
    ...

@dataclass
class TurnResult:
    messages: list[dict]     # 更新后的完整历史(原列表原地修改并返回)
    final_text: str          # 本轮 assistant 最终文字回复
    tool_calls: list[dict]   # 本轮所有工具调用记录(供 trace/折叠块)
    stopped_reason: str      # "" | "max_steps" | "loop_detected"
```

### 5.2 行为

```text
1. messages 已含 [system, ...历史, 刚追加的user]
2. while step < max_steps:
     step += 1
     assistant = _chat_with_retry(client, messages, tools.schemas(), ...)  # 复用
     messages.append(assistant.to_openai_message())
     if not assistant.tool_calls:
         return TurnResult(messages, assistant.content, tool_calls, "")  # 控制权回用户
     tool_msgs, tool_results = _execute_tool_calls(tools, assistant, on_step, step)  # 复用
     for call, res: on_step(step, call.name, call.arguments, res)  # 传 result 给 UI
     messages.extend(tool_msgs)
     循环检测(同现有)
3. return ... stopped_reason="max_steps"/"loop_detected"
```

### 5.3 与现有 `run_agent` 的三处实质差异

1. **接收外部 messages**（不自己建 `[system, user(task)]`）——状态外部化，跨 turn 复用。
2. **`on_step` 多带一个 `result` 参数**——现有是 `(step, name, args)`，这里 `(step, name, args, result)` 才能让 UI 在折叠块里显示工具结果。
3. **不写 trace 文件**——对话式不需要落盘 trace，trace 概念去掉。

### 5.4 复用的子函数（原样从 agent 复制）

`_chat_with_retry`（指数退避）、`_execute_tool_calls`（改签名加 result 回传）、`_freeze`（循环检测签名）、循环检测逻辑。`DeepSeekClient`、`AssistantMessage`、`ToolCall`、`ToolRegistry`、`ToolContext`、`build_tools` 全部原样复用。

### 5.5 offline 模式安置

对话式必须 DeepSeek。offline 时：

- **禁用对话输入**：`st.chat_input` 设 `disabled`，提示"对话需 DeepSeek API key"。
- 不再用 `OfflineClient` 跑对话（它是固定脚本，无法响应对话）。
- `OfflineClient` 在新项目里**删除**，只留 `DeepSeekClient` 单一 client。

## 6. `summarizer` —— 历史超限压缩

### 6.1 触发时机

每次 `run_turn` 前检查：`estimate_tokens(messages) > TOKEN_THRESHOLD` 时触发压缩。

```python
TOKEN_THRESHOLD = 20000   # 留余量,DeepSeek 上下文 ~64k

def estimate_tokens(messages: list[dict]) -> int:
    """粗估 token:中英混合约 1 字≈1 token,空格分隔≈0.25 token。
    不精确,只为触发判断,够用。"""

def summarize_old_messages(
    messages: list[dict],
    client: LLMClient,
    *,
    keep_recent: int = 6,        # 保留最近 N 条消息(3 轮对话)不压缩
    max_retries: int = AGENT_MAX_RETRIES,
) -> list[dict]:
    """把旧消息压缩成一条 system 摘要,保留最近 keep_recent 条。
    返回新 messages:[原system, 摘要system, ...最近keep_recent条]。
    失败时返回原 messages 不变(宁可不压缩也不崩)。"""
```

### 6.2 压缩逻辑

```text
messages = [system, u1, a1, u2, a2, ..., u10, a10]   (超限)
旧段 = messages[1 : -keep_recent]                     # u1..a7
prompt DeepSeek: "把以下对话摘要成要点,保留关键预测结果/数据事实/用户意图,
                   不要编造数值。"
摘要 = client.chat([{system:摘要指令}, {user: 旧段文本}], tools=[])
新 messages = [原system, {role:system, content:摘要}, ...最近keep_recent条]
```

### 6.3 关键设计点

- **保留 system 原文**：不压缩 system（它是 agent 人格/工具说明）。
- **保留最近 keep_recent 条**：最近 3 轮不压缩，保证当前对话连贯。
- **摘要作为新 system 消息注入**：标记为"历史摘要"，不是 user/assistant。
- **失败回退**：DeepSeek 调用失败→返回原 messages，本轮不压缩（宁滥勿崩）。
- **重对象不靠 messages**：`ToolContext` 里的重对象（enso/results）跨 turn 保留，messages 只存文本摘要。与现有架构一致（`tools.py` 的 ToolContext 设计本就为此）。

### 6.4 边界

`keep_recent=6` 是 3 轮（user+assistant 各算一条，工具消息也算）。若单轮工具调用多，实际保留的"轮数"可能少于 3——但够用，不精确控轮数。

## 7. Streamlit chat UI + chat_helpers

### 7.1 `chat_helpers.py`（纯函数，可测）

```python
def init_messages(system_prompt: str) -> list[dict]:
    """新建对话:[{role:system, content:system_prompt}]"""

def append_user(messages: list[dict], text: str) -> list[dict]:
    """追加 user 消息,返回(原地改并返回)"""

def parse_tool_step(step, name, args, result) -> dict:
    """把 on_step 回调参数整理成折叠块渲染用的 dict:
       {step, name, args, result, result_preview(截断)}"""

def should_summarize(messages) -> bool:
    """estimate_tokens > THRESHOLD?"""

def render_hint_no_key() -> str:
    """无 key 时的提示文案"""
```

### 7.2 `app.py`（Streamlit 粘合，无单测，靠手动 smoke）

```text
session_state 初始化:
  - messages = init_messages(SYSTEM_PROMPT)
  - ctx = ToolContext(base_dir=临时目录)
  - tools = build_tools(ctx)
  - client = DeepSeekClient() (无 key 则 None)

主循环:
  - 渲染历史气泡(messages 里 user/assistant/system摘要)
    * assistant 气泡后,若有该轮 tool_calls,渲染折叠块(来自 parse_tool_step)
  - st.chat_input(无 key 时 disabled)
    * 用户输入 → append_user(messages, text)
    * 若 should_summarize → summarize_old_messages
    * run_turn(messages, tools, client, on_step=往气泡流插折叠块)
    * 渲染本轮 assistant 文字 + 工具折叠块
  - sidebar:API key 输入(空则读环境变量)、清空对话按钮
```

### 7.3 折叠块渲染

每个工具调用：一个 `st.expander(f"🔧 step {n}: {tool_name}")`，展开显示参数 JSON + 结果（截断）。在 assistant 最终文字气泡**之前**按顺序出现。

## 8. 错误处理与边界

| 场景 | 处理 |
|---|---|
| 无 API key | `chat_input` disabled，提示"需 DeepSeek key"，不崩 |
| DeepSeek 401/网络错 | `_chat_with_retry` 复用退避；最终失败 `run_turn` 抛 `DeepSeekError`，`app.py` 捕获显示在气泡 |
| 工具抛异常 | `ToolRegistry.execute` 已捕获返字符串（现有），进折叠块，LLM 自行处理 |
| 单轮工具失控 | `max_steps` 上限 + 循环检测（复用）→ `stopped_reason` |
| 历史超 token | `summarize_old_messages` 压缩；压缩失败回退原 messages |
| 临时目录污染 | 每会话一个 `tempfile.mkdtemp`，`atexit` 清理（复用现有 `_session_base_dir` 思路） |
| 用户清空对话 | sidebar 按钮 → `session_state.messages = init_messages(...)` 重置 |

不变量：对话永不因单点失败永久卡死——任一轮失败都能让用户继续发新消息（失败信息进气泡，历史仍可用）。

## 9. 测试

| 文件 | 测试 |
|---|---|
| `test_run_turn.py` | 用 fake client（scripted AssistantMessage）测：① 单轮无工具→返回文字 ② 单轮调一次工具→返回 ③ 连续调工具到不调 ④ max_steps 上限 ⑤ 循环检测 ⑥ messages 接收外部并原地更新 ⑦ on_step 带 result 回调触发 |
| `test_summarizer.py` | ① estimate_tokens 粗估合理 ② should_summarize 阈值 ③ summarize 保留 system+最近 N 条 ④ 压缩成 system 摘要 ⑤ DeepSeek 失败回退原 messages |
| `test_chat_helpers.py` | init_messages / append_user / parse_tool_step / should_summarize 纯函数 |
| `test_tools.py` | 复制+精简后，只测保留的 12 工具（去掉 write_report/build_enso_features 等的测试） |

不写测试：`app.py` Streamlit 组件（同现有项目，只测纯 helper）。fake client 沿用现有 `test_agent.py` 的 `_ScriptedClient` 模式。

## 10. 不包含的内容

本设计不实现：

- 会话持久化（关页面即丢，仅 session_state）。
- `ask_user` 工具主动追问（路线 2，本次不做）。
- 报告生成（`reporting/` 不复制，对话场景不写报告）。
- 真实业务级可预报性模型（沿用现有 lead 分档告警）。
- 多用户/权限/生产部署。
