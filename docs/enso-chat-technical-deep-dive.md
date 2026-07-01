# ENSO 对话式 Agent:关键技术详解

> 本文档对 PPT 大纲(`docs/enso-chat-ppt-outline.md`)中的关键技术点做深度补充,
> 供讲解时展开或答辩时备查。每个技术点配:原理 → 代码实现 → 设计权衡 → 边界与陷阱。
>
> 项目当前规模:~4500 行 Python | 82 测试 | 21 工具 | 双轨预测(baseline/enhanced/CNN-LSTM)+ 三层评估(SODA/Realtime/Persistence)。
> 技术点 1-10 是 agent 工程与基线轨(session 1),11-16 是科学方法升级与实时管道(session 2-5)。

---

## 技术点 1:turn-by-turn loop 的状态外化

### 原理

传统一次性 agent loop(`run_agent`)内部自建 `messages = [system, user(task)]`,跑完返回,状态随函数结束而消亡。对话式 agent 的根本需求是:**状态必须跨函数调用存活**,这样用户下一条消息才能接续上下文。

`run_turn` 的核心设计是**状态外化**:不自己持有 messages,而是接收调用方传入的列表,原地修改并返回。

### 实现

```python
def run_turn(messages, tools, client, *, on_step=None, max_steps=15, loop_limit=3):
    result = TurnResult(messages=messages)   # 引用同一列表,不复制
    step = 0
    while step < max_steps:
        step += 1
        assistant = _chat_with_retry(client, messages, tools.schemas())
        messages.append(assistant.to_openai_message())   # ← 原地追加
        if not assistant.tool_calls:
            result.steps = step
            result.final_text = assistant.content
            return result                                  # ← 控制权还给调用方
        for call in assistant.tool_calls:
            res = tools.execute(call.name, call.arguments)
            if on_step: on_step(step, call.name, call.arguments, res)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": res})
        # 循环检测...
    result.stopped_reason = "max_steps"
    return result
```

### 设计权衡

| 方面 | 选择 | 理由 |
|---|---|---|
| messages 所有权 | 调用方持有,run_turn 借用 | Streamlit 的 `session_state.messages` 是天然持久层;函数无状态 |
| 是否复制 messages | 不复制,原地改 | 避免大列表反复拷贝;返回值同时带引用 |
| 终止条件 | "模型不调工具"即返回 | 一轮对话的语义边界:模型想说话说完了 |
| 控制 UI 渲染 | on_step 回调 | loop 不耦合 Streamlit,纯函数可测 |

### 边界与陷阱

- **messages 必须含 system 在 index 0**:summarizer 依赖此约定(`messages[0]` 当原 system 保留)。
- **tool 消息必须带 `tool_call_id`**:OpenAI/DeepSeek 协议要求 assistant 的 tool_calls 与 tool 结果一一对应,id 匹配,否则 API 报错。
- **原地修改的副作用**:调用方传入的列表会被改变。这是有意的——但若调用方需要保留原始历史(如回滚),需自己先深拷贝。

---

## 技术点 2:三重保护机制(退避 / 循环检测 / 上限)

### 2.1 指数退避重试

**原理**:LLM API 的瞬时错误(429 限流、5xx 服务器、网络抖动)重试通常能成功;但认证错误(401)和参数错误(400)重试无用。需区分 retryable。

**实现**(`client.py`):

```python
except urllib.error.HTTPError as exc:
    retryable = exc.code == 429 or exc.code >= 500   # 仅这两类可重试
    raise DeepSeekError(..., retryable=retryable)
except urllib.error.URLError as exc:
    raise DeepSeekError(..., retryable=True)          # 网络层全可重试
```

`run_turn._chat_with_retry`(`run_turn.py:55`):

```python
attempt = 0
while True:
    try:
        return client.chat(messages, tools)
    except DeepSeekError as exc:
        if not exc.retryable or attempt >= max_retries:
            raise                                    # 不可重试或耗尽 → 抛
        attempt += 1
        delay = min(max_delay, base_delay * (2 ** (attempt - 1)))  # 指数:0.5,1,2,4...
        delay += random.uniform(0, base_delay * 0.1)                # 抖动防雷鸣
        time.sleep(delay)
```

**权衡**:
- `base_delay=0.5, max_delay=8.0, max_retries=3`:首次失败后等 0.5s、1s、2s,最坏总等 3.5s。对课程原型够用;生产可调大。
- 加 jitter(随机 ±5%):防止多个 client 同时重试形成"惊群"。
- `client.chat` 是**纯请求**(不写文件、不改 ctx),所以重试无副作用——这点是重试安全的前提。

### 2.2 循环检测

**原理**:模型有时会卡在重复调用相同工具相同参数的死循环(如反复 `classify_phase(value=0)`)。检测连续相同签名可早停。

**实现**:

```python
def _freeze(arguments): return json.dumps(arguments, sort_keys=True, ensure_ascii=False)

sig = tuple((c.name, _freeze(c.arguments)) for c in assistant.tool_calls)
if sig == last_sig:
    repeat_count += 1
else:
    last_sig = sig
    repeat_count = 1
if repeat_count >= loop_limit:   # 默认 3
    result.stopped_reason = "loop_detected"
    return result
```

**关键细节**:
- `_freeze` 用 `sort_keys=True`:dict 键顺序不同(`{"a":1,"b":2}` vs `{"b":2,"a":1}`)不影响检测,只比语义。
- 比的是**整轮所有 tool_calls 的集合签名**,不是单个调用——模型若一轮里调 A 又调 B 再调 A 又调 B,签名相同,会被检测。
- `loop_limit=3`:连续 3 轮相同才停,给模型一点自我纠正空间(如第 2 轮换个参数就重置计数)。

### 2.3 max_steps 硬上限

**原理**:兜底保护。无论模型怎么跑,单轮内最多 `AGENT_MAX_STURNS=15` 次工具调用,防止 token 失控或逻辑死循环。

**权衡**:15 是对话场景的经验值。一次性 pipeline runner 用 25(要跑完整报告流水线);对话单轮通常 1-3 次工具调用,15 足够宽松又防失控。

---

## 技术点 3:工具层——LLM 与科学引擎的桥梁

### 3.1 工具的抽象结构

每个工具是一个 `Tool` dataclass,四个要素:

```python
@dataclass
class Tool:
    name: str                    # LLM 调用时用的名字
    description: str             # LLM 据此判断何时用
    parameters: dict             # JSON Schema,约束参数
    fn: Callable[..., str]       # 实际执行,返回字符串

    def to_openai_schema(self):  # 转成 DeepSeek function-calling 格式
        return {"type": "function", "function": {"name":..., "description":..., "parameters":...}}
```

### 3.2 ToolRegistry 的错误处理

**原理**:工具执行失败不能让整个 loop 崩——要把异常转成字符串回传给 LLM,让模型自己决定下一步。

**实现**(`tools.py:108`):

```python
def execute(self, name, arguments):
    if name not in self._tools:
        return f"Error: unknown tool '{name}'. Available: {available}"
    tool = self._tools[name]
    try:
        return tool.fn(**arguments)
    except Exception as exc:   # noqa: BLE001 — 故意宽泛
        return f"Error executing tool '{name}': {exc.__class__.__name__}: {exc}"
```

**设计权衡**:
- `except Exception` 宽泛捕获:工具可能抛 ValueError(KeyError/类型错误),全转字符串。代价是隐藏 bug——但 agent loop 稳定性优先。
- 错误字符串回传 LLM:模型看到 `Error: ...` 通常会换参数重试或换工具,形成自我修复。
- unknown tool 也返字符串而非抛:模型有时幻觉出不存在的工具名,返"Available: ..."列表引导它选对的。

### 3.3 "重对象在 ctx,文本在 messages" 的分离

这是工具层最重要的工程决策。

**问题**:LLM 上下文有 token 上限(~64k)。若把 540 行 DataFrame 序列化进 messages,单次对话就爆。

**解决**:工具返回**紧凑字符串摘要**(路径 + 关键数字),重对象存 `ToolContext`:

```python
@dataclass
class ToolContext:
    enso: pd.DataFrame | None        # 540 行,LLM 看不见
    results: dict | None             # 完整结果 dict
    predictions: pd.DataFrame | None
    figure_paths: list[Path]         # 生成图路径
    enso_data_source: str | None     # 缓存键

def _load_enso_data(ctx, ...):
    output = run_enso_forecast(...)
    ctx.enso = output.enso           # 重对象进 ctx
    ctx.results = output.results
    return f"rows=540, date=1980→2024, best=linear_ridge..."  # LLM 只看摘要
```

**收益**:
- messages 体积小,token 省,多轮对话不爆。
- 重对象跨轮复用:`forecast_for_month` 第二次调同 lead,直接读 `ctx.results["latest_forecast"]`,不重训。
- 幂等性:`load_enso_data` 检查 `ctx.results is not None and ctx.enso_data_source == data_source`,命中走缓存。

---

## 技术点 4:lead 换算与可信度分档

### 原理

ENSO 预测的核心物理量是 **lead(提前量)**:从最新数据月到目标月的月数。可预报性随 lead 衰减——这是 ENSO 动力学的硬约束,不是技术限制。

### 实现(`tools.py:_forecast_for_month`)

```python
def _compute_lead(last_date, target_year, target_month):
    return (target_year - last_date.year) * 12 + (target_month - last_date.month)

def _forecast_for_month(ctx, target_year, target_month, data_source="auto"):
    # 自动加载(若未加载)
    if ctx.enso is None: _load_enso_data(ctx, data_source=data_source)
    last_date = pd.Timestamp(ctx.enso["date"].max())
    lead = _compute_lead(last_date, target_year, target_month)

    if lead <= 0:        return "目标月已过去,无需预测"
    if str(lead) in ctx.results["latest_forecast"]:
        return cached_result                              # 1/3/6 复用
    if lead >= 12:       return HARD_WARNING              # ≥12 拒绝
    fc = _forecast_value_for_lead(ctx, lead)              # 2/4/5/7-11 现训
    if lead >= 7: tag = "[低可信度]"
    return f"value={fc['value']}, phase={fc['phase']}, ..."
```

### 分档表

| lead | bucket | 行为 | 物理依据 |
|---|---|---|---|
| ≤0 | past | "无需预测" | 目标月已被数据覆盖 |
| 1/3/6 | cached | 复用已训结果 | `load_enso_data` 已训这三个 lead |
| 2/4/5 | short | 现训,正常可信 | 短期可预报性高 |
| 7-11 | low_confidence | 现训,标低可信 | ENSO 可预报性 ~6 个月后衰减 |
| ≥12 | out_of_range | **拒绝预测** | 超出可靠范围,硬告警 |

### 设计权衡

- **为何 ≥12 直接拒绝而非给数字**:ENSO 预测 12 个月后相关系数通常 <0.3,给数字会误导。系统宁可说"超出可靠范围,建议刷新数据或改近月",这是科学诚实。
- **为何 7-11 仍预测但标低可信**:边界灰区,用户可能需要参考。给数字但明确标注"indicative only",决策权交用户。
- **临时 lead 的模型选择**:`_forecast_value_for_lead` 对临时 lead 没有 test split 评估,直接选 random_forest(在 sample 数据上通常最优)。这是已知简化,代码注释和 `recommend_data_range` 都如实说明。

> **演进**:技术点 11 介绍了增强轨的**数据驱动分档**——用 per-lead ACC 替代这里的硬编码 7/12 阈值,仅在 enhanced 路径局部应用(不破坏基线测试)。

---

## 技术点 5:特征工程与防数据泄露

### 原理

时间序列预测的最大陷阱是**数据泄露**(data leakage):用未来信息训练,导致测试集表现虚高,实际部署崩盘。ENSO 预测必须严格只用历史。

### 实现(`enso_features.py`)

```python
def make_enso_supervised_table(df, leads=(1,3,6), max_lag=12):
    # 特征:只用当前和历史(lag≥0 是当前,lag>0 是历史)
    for lag in range(max_lag + 1):           # lag=0,1,...,12
        data[f"nino34_lag_{lag}"] = data["nino34"].shift(lag)

    # 滚动均值:只用当前及之前(min_periods 防开头 NaN)
    data["roll_mean_3"] = data["nino34"].rolling(3, min_periods=1).mean()
    data["roll_mean_6"] = data["nino34"].rolling(6, min_periods=1).mean()

    # 季节周期:月份的 sin/cos 编码
    data["month_sin"] = np.sin(2*np.pi*month/12)
    data["month_cos"] = np.cos(2*np.pi*month/12)

    # 目标:未来 h 个月(负号 shift = 往后看)
    for lead in leads:
        data[f"target_lead_{lead}"] = data["nino34"].shift(-lead)   # ← 关键:负号
```

### 防泄露的三道防线

1. **特征只用 lag≥0**:`shift(0)` 是当前值,`shift(1)` 是上月——全是已发生的。绝不用 `shift(-k)`(未来)做特征。
2. **目标用 `shift(-lead)`**:负号表示"未来第 lead 个月的值"。这是要预测的,不是特征。
3. **时间顺序切分**(`evaluation.py`):

```python
def temporal_train_test_split(df, test_fraction=0.2):
    split_index = int(round(len(df) * (1.0 - test_fraction)))
    train = df.iloc[:split_index]    # 前面训练
    test = df.iloc[split_index:]     # 后面测试
```

**绝不用 `sklearn.train_test_split`**(随机打乱)——那会让测试集混入训练集之前的时间点,形成时间泄露。

### 边界

- `rolling(min_periods=1)`:开头几行窗口不满,用 `min_periods=1` 避免产生 NaN(否则 dropna 丢数据)。
- `max_lag=12`:一年滞后,覆盖 ENSO 年际信号;再长边际收益低且增加维度。

---

## 技术点 6:历史摘要压缩

### 原理

DeepSeek 上下文窗口 ~64k token。多轮对话后,历史累积会逼近上限。直接截断丢信息,完整保留会爆。折中:**旧消息压缩成摘要,最近消息保留原文**。

### 实现(`summarizer.py`)

```python
def summarize_old_messages(messages, client, *, keep_recent=6):
    if len(messages) <= 1 + keep_recent:
        return messages                              # 太短不压缩

    original_system = messages[0]                    # 保留 system 提示词
    old = messages[1 : len(messages) - keep_recent]  # 旧段
    recent = messages[len(messages) - keep_recent :] # 最近 N 条

    old_text = "\n".join(f"[{m['role']}] {m['content']}" for m in old)
    summary = client.chat(
        [{"role":"system", "content": "摘要要点,保留预测结果,不编造数值"},
         {"role":"user", "content": old_text}],
        tools=[], tool_choice="none"
    ).content

    return [original_system,
            {"role":"system", "content": summary},   # 摘要作新 system
            *recent]
```

### 设计权衡

| 决策 | 选择 | 理由 |
|---|---|---|
| 触发判断 | `estimate_tokens > 20000` | 留 3 倍余量(64k→20k 触发),防边界 |
| token 估算 | 数字符 | 粗糙但够用;精确需调 tokenizer,过度工程 |
| 保留多少 | keep_recent=6 | 约 3 轮对话,保证当前上下文连贯 |
| 摘要放哪 | 新 system 消息 | 标记为历史上下文,不与原 system 混淆 |
| 失败处理 | 返回原 messages | 宁可不压缩也不崩 |

### 边界与陷阱

- **工具消息也在压缩范围**:old 段可能含 `role:tool` 的结果,被一起摘要。这是对的——工具结果的关键信息(数值、阶段)应进摘要,细节丢弃。
- **重对象不靠 messages**:`ToolContext` 的 enso/results 跨压缩仍存活——它们在 ctx,不在 messages。压缩只动文本历史。
- **`tool_choice="none"`**:摘要调用不让模型调工具,纯文本生成。
- **`keep_recent=6` 不精确控轮数**:若一轮有多个 tool 消息,6 条可能不足 3 轮。但够用,不精确控。

---

## 技术点 7:LLM 客户端的协议设计与重试分类

### LLMClient Protocol

```python
class LLMClient(Protocol):
    def chat(self, messages, tools, tool_choice="auto") -> AssistantMessage: ...
```

**原理**:Protocol(结构化子类型)让任何实现该方法的类都算 client,无需继承。`DeepSeekClient` 实现它;测试用 `_ScriptedClient` 也实现它——loop 不知道也不关心具体是哪个。

### DeepSeekClient 的请求构造

```python
payload = {
    "model": "deepseek-chat",          # 必须是 chat;reasoner 不支持 function calling
    "messages": messages,
    "tool_choice": tool_choice,        # "auto"=模型自选 / "none"=不调工具
}
if tools: payload["tools"] = tools     # 空工具列表不发(省 token)
```

**关键**:DeepSeek 的 `deepseek-reasoner`(推理模型)不支持 function calling,`_resolve_deepseek_config` 显式拒绝:

```python
if mdl == "deepseek-reasoner":
    raise DeepSeekError("deepseek-reasoner does not support function calling. "
                        "Use deepseek-chat for the agentic tool loop.")
```

### 错误分类表

| HTTP | 类型 | retryable | 处理 |
|---|---|---|---|
| 429 | 限流 | True | 退避重试 |
| 5xx | 服务器 | True | 退避重试 |
| URLError | 网络/DNS/超时 | True | 退避重试 |
| 401/403 | 认证 | False | 立即抛 |
| 400/404 | 参数/路径 | False | 立即抛 |
| 无 choices | 响应解析 | False | 立即抛 |

**设计依据**:`client.chat` 是纯请求(无副作用),所以重试安全。若 client 内部有状态(如游标),重试需更谨慎。

---

## 技术点 8:用户 CSV 上传的完整链路

### 数据流

```
[浏览器] file_uploader 选 my_enso.csv
    ↓ uploaded.getvalue() (bytes)
[_handle_uploaded_csv] 存到 {base_dir}/data/user/my_enso.csv
    ↓ 路径写进 session_state["user_csv_path"]
[用户对话] "用我上传的数据"
    ↓ append_user(messages, ...)
[LLM 决定] 调 load_user_enso(path=".../my_enso.csv")
    ↓
[load_enso_csv] 校验:date + nino34 列必须存在,否则 ValueError
    ↓ 校验行数 ≥ 30(否则数据太少无法训练)
[run_forecast_on_enso] 特征→训练→1/3/6 lead→结果
    ↓ ctx.enso ← 用户数据
    ↓ ctx.results ← 新模型结果
[后续工具] forecast_for_month / 画图 自动用用户数据
```

### `run_forecast_on_enso` 的提取(关键重构)

**问题**:原 `run_enso_forecast` 把"加载数据"和"跑预测"耦合在一起——它内部调 `_resolve_enso_data`(sample/NOAA)。用户 CSV 走不了这条路。

**解决**:提取核心预测逻辑成独立函数,两种加载方式共用:

```python
def run_forecast_on_enso(enso, *, outputs_dir, data_source_info):
    """已有 enso DataFrame → 跑预测。与数据来源无关。"""
    table, feature_cols = make_enso_supervised_table(enso, ...)
    train, test = temporal_train_test_split(table, ...)
    # ... 训练 + 评估 + 最新预测 ...
    return results, results_path, predictions_path

def run_enso_forecast(base_dir, data_source, refresh_noaa):
    enso, info = _resolve_enso_data(...)              # sample/NOAA 加载
    return run_forecast_on_enso(enso, ...)            # 调核心

def _load_user_enso(ctx, path):
    enso = load_enso_csv(path)                        # 用户 CSV 加载
    results, *_ = run_forecast_on_enso(enso, ...)     # 调同一核心
```

**收益**:DRY——预测逻辑只有一份,两种数据源共享。加第三种数据源(如 GPCP 降水)只需新加载函数 + 调 `run_forecast_on_enso`。

### 校验与错误处理

```python
def _load_user_enso(ctx, path):
    csv_path = Path(path)
    if not csv_path.exists():
        return f"Error: file not found: {path}"
    try:
        enso = load_enso_csv(csv_path)     # 缺列抛 ValueError
    except ValueError as exc:
        return f"Error: {exc}"
    if len(enso) < 30:
        return f"Error: only {len(enso)} rows; need ≥30 (2+ years)"
    # ... 成功路径
```

**设计**:所有失败路径返 Error 字符串(不抛),ctx 不被修改——`ctx.enso` 保持原值或 None。这让 agent 能看到错误并建议用户重传。

---

## 技术点 9:科学引擎的模型选择与评估

### 三类模型

```python
def build_model_suite(random_state=42):
    return {
        "linear_ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "random_forest": RandomForestRegressor(n_estimators=120, max_depth=8,
                                                min_samples_leaf=3, random_state=42),
    }
# 加上 persistence baseline(未来=现在),共 3 类
```

| 模型 | 角色 | 超参选择理由 |
|---|---|---|
| Persistence | 基线 | 判断 ML 是否真有效;若 ML 不如它,说明特征/模型有问题 |
| Ridge(α=1.0) | 线性可解释 | α=1.0 是温和正则;StandardScaler 防特征尺度差 |
| RandomForest | 主 ML | 120 树够稳;max_depth=8 防过拟合;min_samples_leaf=3 防叶节点过纯 |

### 评估指标

```python
def calculate_regression_metrics(y_true, y_pred):
    rmse = sqrt(mean((y_true - y_pred)**2))   # 均方根误差,量纲同原值
    mae = mean(|y_true - y_pred|)             # 平均绝对误差,抗离群
    corr = corrcoef(y_true, y_pred)[0,1]      # 相关系数,方向性
    return {"rmse":..., "mae":..., "corr":...}
```

**为何用 RMSE 选最佳模型**:RMSE 对大误差敏感(平方放大),符合预测场景(宁愿多次小错也不要一次大错)。MAE 抗离群但不够敏感,corr 只看方向不看幅度。

### 最新预测的 fit_models_for_latest_forecast

```python
def fit_models_for_latest_forecast(models, table, feature_cols, lead):
    latest_features = table.iloc[[-1]][feature_cols]   # 最后一行=最新数据
    for name, model in models.items():
        model.fit(table[feature_cols], table[target])  # 全量训练
        forecasts[name] = model.predict(latest_features)[0]  # 预测未来
    forecasts["persistence"] = table.iloc[-1]["nino34_lag_0"]  # 基线=当前值
```

**注意**:评估用 train/test split,但**最新预测用全量数据训练**(test split 是为评估指标,实际预测要用所有可用数据)。这是正确做法——评估回答"模型准不准",最新预测回答"未来是多少"。

---

## 技术点 10:Streamlit chat 的会话状态管理

### session_state 的角色

Streamlit 每次交互(点按钮、输入)都**重跑整个脚本**。要跨重跑保持状态,必须存 `session_state`。

```python
# 初始化(仅首次)
if "messages" not in st.session_state:
    st.session_state["messages"] = init_messages(SYSTEM_PROMPT)
if "ctx" not in st.session_state:
    st.session_state["ctx"] = ToolContext(base_dir=_session_base_dir())
    st.session_state["tools"] = build_tools(st.session_state["ctx"])
if "shown_figures" not in st.session_state:
    st.session_state["shown_figures"] = set()   # 已显示的图,防重复
```

### 三类持久状态

| session_state 键 | 内容 | 生命周期 |
|---|---|---|
| `messages` | 对话历史(list[dict]) | 整个会话;清空按钮重置 |
| `ctx` + `tools` | ToolContext + 工具注册表 | 整个会话;重对象跨轮复用 |
| `shown_figures` | 已显示图路径集合 | 整个会话;防图重复显示 |
| `base_dir` | 临时工作目录 | 整个会话;atexit 清理 |

### 图的增量显示

```python
def _new_figures(ctx):
    shown = st.session_state.setdefault("shown_figures", set())
    fresh = [p for p in ctx.figure_paths if str(p) not in shown]
    shown.update(str(p) for p in fresh)   # 标记已显示
    return fresh

# 在 assistant 气泡内:
for fig in _new_figures(ctx):
    st.image(str(fig), caption=fig.name, use_container_width=True)
```

**原理**:`ctx.figure_paths` 跨轮累积(每画一张就 append)。但 UI 只想显示"本轮新增"的图,不重复显示历史图。用 `shown_figures` 集合记录已显示的,每次只渲染 fresh。

### 边界

- **关页面即丢**:session_state 是内存态,刷新或关闭浏览器后清空。这是设计选择(原型够用),非 bug。
- **临时目录清理**:`atexit.register(shutil.rmtree, path)` 确保进程退出时清理,不污染磁盘。

---

## 技术点 11:数据驱动的 lead 可信度分档(增强轨)

### 原理

技术点 4 的硬编码分档(7/12 月阈值)是经验值,答辩时被问"为什么是 7 不是 8"难答。增强轨改成**读 per-lead ACC 数据驱动分档**:某个 lead 在测试集上 ACC 跌破阈值才降档,有数据支撑。

### 实现(`tools.py:_confidence_from_acc` + `config.py`)

```python
ACC_LOW_CONF = 0.5   # 低于此值标低可信
ACC_REFUSE = 0.3     # 低于此值拒绝预测

def _confidence_from_acc(acc):
    if acc < ACC_REFUSE:
        return "refuse", f" [拒绝: ACC={acc:.2f}<{ACC_REFUSE}]"
    if acc < ACC_LOW_CONF:
        return "low_confidence", f" [低可信: ACC={acc:.2f}<{ACC_LOW_CONF}]"
    return "normal", f" [ACC={acc:.2f}]"
```

### 设计权衡

| 方面 | 选择 | 理由 |
|---|---|---|
| 应用范围 | **仅 enhanced 路径** | 不改 `_forecast_for_month` 硬编码,避免破坏基线 38 项测试和 CNN-LSTM 轨 |
| 阈值可调 | 写进 `config.py` | 0.5/0.3 是经验值,答辩可讨论;不像硬编码藏在函数里 |
| ACC 来源 | 测试集 in-sample | 临时 lead 无独立 test split,用 train/test split 的 ACC 作粗档,诚实标注"optimistic" |

### 边界

- **ACC 是历史技能不是未来保证**:某个 lead 历史 ACC=0.6,不代表这次预测一定准——它只说"这个 lead 上模型平均比随机强"。agent 转述时用 ACC 解释可信度,但仍标"参考性"。
- **阈值 0.5/0.3 的依据**:0.5 是 ENSO 预报界"有技能"的常见门槛(Ham et al. 用);0.3 是"几乎无技能"的经验下限。可调。
- **不全局替换硬编码**:基线 `_forecast_for_month` 仍用 7/12 硬档——因为基线没接外生变量,ACC 曲线不同,且改它会回归。两套分档并存是刻意的工程取舍。

---

## 技术点 12:CNN-LSTM 空间场轨与离线训练/在线推理分离

### 原理

单变量自回归(技术点 9)撞 ENSO 物理天花板(春季预测障碍)。突破要靠空间场前兆信号(sst/t300/ua/va)。CNN-LSTM 是这条轨的模型,对标 Ham et al. 2019 (Nature)。关键工程约束:**训练在 GPU/离线,推理在 CPU/在线,两者彻底分离**——Streamlit 进程永远不训练神经网络。

### 实现(`src/models/cnn_lstm.py` + `scripts/train_cnn_lstm.py`)

```
离线训练(scripts/train_cnn_lstm.py,不进 Streamlit):
  SODA sst/t300/ua/va (100年×36月×24纬×72经)
    → 连续月序列滑窗(12月输入 → 24 lead 目标)
    → 留缓冲三划分(训0-70/验70-82/缓冲82-85/测85-99)
    → 只用训练集统计量标准化
    → CNN(3conv)+LSTM(2层)+FC(24) 训练,早停
    → 权重存 weights/cnn_lstm_soda.pth (49M,含 x_mean/x_std)

在线推理(predict_cnn_lstm,torch 懒加载):
  输入窗口 (12,24,72,4) → 标准化(用 checkpoint 的 SODA 统计量)
    → model.forward() → 24 lead {value, phase}
```

### 设计权衡

| 方面 | 选择 | 理由 |
|---|---|---|
| 训练/推理分离 | 离线脚本 + 内置权重 | Streamlit 无 GPU、要秒级响应;训练耗时数十分,不该在线做 |
| torch 依赖 | 懒加载(工具内 import) | `import tools` 不触发 torch;仅 `forecast_cnn_lstm` 调用时才加载 |
| 划分留缓冲 | 82-85 年不用 | 测试窗口起点≥85,与训练尾隔开,防时序渗透虚高 ACC |
| 标准化 | 只用训练集统计量,存 checkpoint | 推理时复用训练分布;若用全集统计量=测试集泄露 |
| 数据 | SODA-only(无 CMIP) | 仓库无 CMIP 数据;走 notebook 默认路径,小样本靠早停+dropout 缓解 |

### 边界与陷阱

- **小样本过拟合**:SODA 100 年单一再分析,CNN-LSTM 易过拟合。Dropout(0.7)+weight_decay+早停缓解,但这是已知局限,答辩"局限性"页写明。
- **SODA 匿名年份**:SODA 的 year/month 是竞赛匿名索引(1-100 block),非真实日历。这导致无法做 SODA-vs-realtime 同月对比(见技术点 14),但不影响训练(训练只用相对时序)。
- **t300 vs 热含量**:论文 Ham et al. 用上层 300m 热含量(HC),本轨用 t300(300m 层温度),信息量弱一档——这是 SODA 变量所限,已知简化。
- **权重体积**:49M 内置仓库偏大,可接受;若云部署要省体积可后续换 onnxruntime。

---

## 技术点 13:数据源注册表与外生指数实时拉取

### 原理

"让 agent 自主找数据"若做成自由 web 搜索,可靠性差且不可复现。本项目的取舍:**预置受控数据源注册表**,agent 在审定过的源里列/选/加载,有自主体感又不失控。

### 实现(`src/data/source_registry.py`)

```python
@dataclass(frozen=True)
class DataSource:
    name: str; description: str; url: str; value_col: str; coverage: str

REGISTRY = {
    "nino34": DataSource(..., "https://psl.noaa.gov/.../nino34.long.anom.data", ...),
    "soi":    DataSource(..., ".../soi.long.data", ...),    # 大气端前兆
    "nino12": DataSource(..., ".../nino12.long.data", ...), # 东太平洋海洋端前兆
}
# 复用 noaa_enso.parse_nino34_table 的解析逻辑,泛化为 parse_year_month_table
# 三个源同格式(年+12值,-99.99缺测),解析器一份共用
```

工具层:`list_data_sources`(列源) / `load_index(name)`(拉取+缓存) / `forecast_enhanced`(自动用 soi+nino12)。

### 设计权衡

| 方面 | 选择 | 理由 |
|---|---|---|
| 自由搜索 vs 注册表 | 注册表(受控) | ENSO 预测对数据质量极敏感,自由抓取=不可控;注册表保证格式/单位/覆盖已知 |
| WWV 替代 | 用 Niño1+2 | WWV 无可靠静态源(CPC 不可达、PMEL 文件移除);Niño1+2 是东太平洋上涌区,ENSO 发展前兆,与 Niño3.4 区域不同有独立信息 |
| 解析器复用 | 泛化 `parse_year_month_table` | 三源同格式,一份解析器;加新源只需注册一条 |
| 缓存 | raw/processed 双文件 | 复用 noaa_enso 模式;首次下载,后续走缓存 |

### 边界

- **Niño3.4 URL 修复**:原 `gcos_wgsp/Timeseries/...` 路径已 404(NOAA/PSL 迁移),修成 `data/timeseries/month/data/...`。这是隐藏 bug——靠 sample 回退没暴露,但"真要能用"必须修。
- **SOI 末尾缺测**:SOI 序列末尾几个月常 -99.99,merge on date 自动剔除。
- **不做 web search 发现**:自由发现(搜候选源、自动试探接入)是后续可选项,本阶段不做——可靠性优先于"全自动"噱头。

---

## 技术点 14:实时空间场管道与跨域异常化对齐

### 原理

CNN-LSTM 训练在 SODA(存 anomalies,mean≈0),推理却喂 OISST/GODAS/NCEP(存绝对值)。直接喂=致命域偏移。**异常化对齐**是核心:每个实时源减自身 30 年月气候态转异常,匹配 SODA 的异常分布,再用 SODA 训练统计量标准化。

### 实现(`src/data/realtime_fetch.py` + `climatology.py`)

```
四通道实时拉取(全免注册 OPeNDAP):
  sst  : NCEI OISST 日值(月中1天近似) → 月均 → 异常化 → 重采样
  t300 : PSL GODAS pottmp(303m层,K→°C) → 异常化 → 重采样
  ua/va: PSL NCEP/NCAR R1(850hPa) → 异常化 → 重采样
  ↓
窗口统一截止到风场最新月(~5月滞后是瓶颈),诚实标注不伪造填充
  ↓
predict_cnn_lstm_realtime: 用 SODA 的 x_mean/x_std 标准化 → CNN 前向
  ↓
结果标注 "cross-domain, 精度低于 SODA hindcast"
```

### 设计权衡

| 方面 | 选择 | 理由 |
|---|---|---|
| 跨域处理 | 异常化对齐(不重训) | 重训需建新训练管道,工程翻倍;异常化缓解域偏移够用,诚实标注剩余损失 |
| 风场滞后 | 接受 5 月,窗口截止风场最新月 | 换 ERA5 需注册;5月滞后可接受,不拿旧风场填充假装实时 |
| SST 月均 | 月中1天近似 | 真月均要下30个日文件(336请求,慢且脆);1天近似对 ENSO 月值精度够 |
| 大文件 | OPeNDAP 切片 | NCEP 整文件437MB下载断连;OPeNDAP 按时间/层切片几秒拉完 |
| 3D interp | 逐时间片 | OPeNDAP 对3D整段interp返回0;逐2D切片resample正确 |

### 边界与陷阱(全是实测踩过的)

- **OPeNDAP 3D interp 返回 0**:`da.interp(lat,lon)` 对含 time 的 3D 整段请求,服务器返回全 0。必须先 `isel(time=i)` 取 2D 片再 interp。
- **GODAS `interp(level=300)` 远程返回 0**:OPeNDAP 不支持跨层 interp,改 `sel(level=303, nearest)`,偏差 3m 可忽略。
- **netCDF4 中文路径**:读写 `D:\作品\...` 都失败(PermissionError/FileNotFoundError)。读用临时 ASCII 拷贝;气候态改 npz 存储。
- **OISST sst 4 维**:有 zlev 层,需 `isel(zlev=0)` squeeze。
- **陆地 NaN**:重采样到海洋外陆地点产生 NaN,`nan_to_num` 填 0,否则 CNN 输出 NaN。
- **跨域精度不可用 SODA ACC 评估**:这是诚实底线——SODA 域 ACC 0.77 不能说 realtime 也 0.77。见技术点 15。

---

## 技术点 15:三层评估协议(SODA / Realtime / Persistence)

### 原理

"预测准不准"不能靠感觉,要用 hindcast(历史回测)算每个 lead 的 ACC。但训练域(SODA)和推理域(realtime)不同,ACC 不能跨域套用。所以建**三层评估**:SODA hindcast(方法上限)、Realtime hindcast(真实跨域效果)、Persistence 基准(技能锚,两域都有)。

### 实现(`src/models/hindcast.py` + `realtime_hindcast.py`)

```
SODA hindcast:
  SODA 测试集(85-99年) → CNN 前向 → 对比 SODA 真实 nino → per-lead ACC
  + Persistence 基线(forecast = 最后观测月)
  → 可靠窗口 lead4-23, SPB 低谷可见, 对标 Ham et al.

Realtime hindcast:
  OISST/GODAS/NCEP 历史段(2020-2021) → 异常化(用 2005-2015 气候态,不重叠防泄露)
  → CNN 前向 → 对比 NOAA 真实 Niño3.4 → per-lead ACC + Persistence
  → lead1-2 跨域可靠(0.86/0.75), 中长 lead 待扩样本
```

### 设计权衡

| 方面 | 选择 | 理由 |
|---|---|---|
| 指标 | all-season ACC | Ham et al. 2019 口径;ENSO 预报标准技能分 |
| 基准 | Persistence | 最简单可行基线(未来=现在);CNN 必须跑赢它才有存在价值 |
| 两域独立评估 | SODA + Realtime 各算 | 跨域 ACC 不可迁移,必须分别评 |
| 泄漏防护 | 气候态期 ≠ 评估期 | realtime 用 2005-2015 算气候态、2020-2021 评估,不重叠 |
| 工具化 | `report_hindcast_skill` / `report_realtime_skill` | agent 能自主调工具回答"准不准",这是项目特色 |

### 边界

- **SODA 匿名年份限制**:SODA year 是竞赛索引,无法映射真实日历。导致 SODA-vs-realtime 同月对比不可行(曾尝试做"跨域差异诊断"失败,已删脚本)。但不影响各自 hindcast——SODA 用自己时序,realtime 用真实日历。
- **Realtime n=7 样本小**:评估窗 2020-01~2021-06 只 7 个窗口,中长 lead ACC 不稳(出现 Persistence 0.9+ 异常值)。短 lead 结论可信,中长待扩到 ~30 窗口(2018-2021)。
- **ACC 是相关性不是精度**:ACC 高只说"相位对",不代表数值准。RMSE 补幅度信息,但选模用 RMSE、报技能用 ACC(ENSO 相位比幅度重要)。
- **agent 路由**:两个 hindcast 工具描述需明确区分,避免 realtime 问句误调 SODA 工具。已知优化点。

---

## 技术点 16:对话驱动的科学决策(agent 自主编排)

### 原理

本项目最终价值不是"预测更准",而是"agent 能自主调度多方法 + 自评可靠性"。一次用户提问,agent 不被动应答,而是自主编排:加载合适数据 → 选方法 → 预测 → 调评估工具查技能 → 诚实给建议。这是"对话驱动科学决策"的完整闭环。

### 实现(真实 DeepSeek 对话轨迹)

用户:"明年3月的Niño3.4会是多少?请用增强方法预测,并和CNN-LSTM对比"

```
agent 自主 7 步编排(无人工干预):
  step1 load_enso_data(auto→真实NOAA, 1876行, 到2026-04)
  step2 load_index(soi)        ← 在线拉外生指数
  step3 load_index(nino12)     ← 在线拉外生指数
  step4 recommend_data_range(2027-03 → lead11, 低可信桶)
  step5 forecast_enhanced(2027-03) → value=0.03, ACC=0.38 → 自动标低可信
  step6 forecast_cnn_lstm(realtime, lead11) → 跨域标注
  step7 forecast_latest(lead1) → 主动补短期 La Niña
  → 生成三方法对比表 + "建议6月内再关注" 的可靠性建议
```

### 设计权衡

| 方面 | 选择 | 理由 |
|---|---|---|
| 自主编排 vs 脚本回放 | 自主(LLM 决策) | OfflineClient 已删;对话式必须 LLM 理解意图选工具,这是 agent 价值所在 |
| 多方法并存 | 三槽不互相覆盖 | `results`/`enhanced_results`/`cnn_forecasts` 分槽,同会话对比不丢数据 |
| 可靠性自评 | 工具化(agent 调 hindcast) | 不在 prompt 里编"我相信这个预测",而是调工具拿 ACC 数字,有据可查 |
| 诚实建议 | agent 用 ACC 解释 + 给替代方案 | "lead11 ACC=0.38 低可信,建议6月内再关注"——不是敷衍,是数据驱动 |

### 边界

- **agent 路由不完美**:有时 realtime 问句会误调 SODA hindcast 工具(两工具描述相似)。这是 LLM 路由优化项,非代码 bug——工具本身工作正常。
- **编排深度依赖 prompt**:`SYSTEM_PROMPT` 引导"如实说明不确定性,不编造数值"。换更弱模型可能编排退化。
- **不是全自动科研**:agent 在审定工具集内决策,不发明新方法。这是刻意的——可靠性优先于"全自动"噱头。评委会问"agent 能自己加方法吗",诚实答"不能,加方法是人定义工具,agent 在工具集内自主调度"。

---

## 附:技术点与 session 对应

| 技术点 | session | 主题 |
|---|---|---|
| 1-10 | session 1 | agent 工程 + 基线轨(loop/工具/状态/防泄露/客户端/CSV/评估/Streamlit) |
| 11 | session 4 | 数据驱动 lead 分档(增强轨) |
| 12 | session 2 | CNN-LSTM 空间场轨 + 离线训练/在线推理 |
| 13 | session 4 | 数据源注册表 + 外生指数 |
| 14 | session 5 | 实时空间场管道 + 跨域异常化 |
| 15 | session 3+5 | 三层评估协议(SODA/Realtime/Persistence) |
| 16 | session 5 | 对话驱动科学决策(主线收束) |
