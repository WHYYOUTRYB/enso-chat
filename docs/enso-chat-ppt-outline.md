# ENSO 对话式 Agent:实现与原理

> PPT 大纲 + 每页讲稿。重点放在"如何实现"和"背后原理"。
> 
> 项目:enso-chat/ | 代码 2296 行 Python | 26 个测试 | 14 个工具

## 第 1 页:标题页

**标题**:ENSO 对话式预报 Agent
**副标题**:从"一次性流水线"到"全对话式"的架构演进
**脚注**:○○○ 课程项目 | 2026-07

**讲稿**:大家好。今天讲一个对话式 ENSO 预报 agent 的实现和原理。ENSO 是厄尔尼诺-南方涛动,热带太平洋最重要的海气耦合模态。这个 agent 能和你自然对话——你说"明年 3 月怎么样",它自动算提前量、加载数据、预测、画图、甚至可以上传自己的数据。我从零开始讲清楚每一层怎么实现。

## 第 2 页:系统能做什么

**标题**:系统能力一览

- 对话式预测:自然语言提需求月,agent 自动算 lead 并预测
- 可信度分档:1-6 月正常,7-11 月低可信,≥12 月拒绝预测
- 三种建模方法:Persistence 基线 / Ridge 回归 / Random Forest
- 6 类工具:数据加载 / 预测诊断 / 画图 / 降水分析 / 潮汐演示 / CSV 上传
- 上传自有数据:上传 ENSO CSV,agent 自动建模
- 对话历史管理:太长自动摘要压缩

**讲稿**:先给一个整体印象。用户打开 Streamlit 聊天页,可以问任何 ENSO 相关问题。agent 不只是查表——它现场训练机器学习模型、做预测、画图、还会根据提前量判断结果可不可信。如果你传了自己的 Niño3.4 数据,它会用你的数据替换示例数据。

## 第 3 页:整体架构

**标题**:三层架构:引擎 → 工具 → 对话 loop

```
┌────────────────────────────────┐
│  Streamlit chat UI (app.py)   │  ← 用户交互层
├────────────────────────────────┤
│  run_turn (turn-by-turn loop) │  ← 对话调度层
│  + summarize_old_messages     │     (新写,非一次性 loop)
├────────────────────────────────┤
│  14 个工具 (tools.py)         │  ← 工具层
│  ToolContext (共享状态)       │     每个工具返回字符串
├────────────────────────────────┤
│  科学引擎                      │  ← 模型层
│  data/features/models/分析/画图 │     纯 Python,无 LLM 依赖
└────────────────────────────────┘
```

**讲稿**:这张图是核心。整系统分四层。最底下是科学引擎——数据处理、特征工程、机器学习模型、可视化,全部纯 Python,pandas + scikit-learn,和 LLM 无关。工具层把引擎的每个功能包装成"工具"——每个工具有名字、描述、JSON Schema 参数、一个可调用函数,返回紧凑字符串而非 DataFrame,让 LLM 能高效消费。对话调度层是新写的 `run_turn`——它不是一次性跑完,而是每轮"模型回复→调工具→喂回结果→直到模型不调工具"就返回控制权,等用户下一条消息。最上层是 Streamlit chat,纯粘合代码。

## 第 4 页:工具层设计——让 LLM 能"动手"

**标题**:14 个工具 = 14 个 LLM 可调用的"手"

| 工具 | 做什么 | 原理要点 |
|---|---|---|
| `load_enso_data` | 加载 sample/NOAA 数据并训练模型 | 幂等:同源二次调用走缓存 |
| `load_user_enso` | 加载用户上传的 ENSO CSV | 调 `run_forecast_on_enso` 核心 |
| `forecast_for_month` | 预测指定目标月 Niño3.4 | lead 分档:≤6 正常,7-11 低可信,≥12 拒 |
| `classify_phase` | Niño3.4 值→El Niño/La Niña/Neutral | ±0.5℃ 阈值 |
| `diagnose_local_data` | 诊断本地数据覆盖与新鲜度 | 纯 FS 扫描,零 API |
| `recommend_data_range` | 评估目标月是否在可靠范围内 | 返 bucket + allow_run |
| 4 个画图工具 | 时间序列/预测对比/RMSE/阶段图 | matplotlib → PNG |
| `analyze_precipitation` | 降水异常按 ENSO 阶段统计 | boxplot |
| `run_tide_prediction` | 潮汐演示预测 | Ridge + 调和特征 |

**讲稿**:每个工具就是 LLM 的一只"手"。关键设计就一个原则:**工具返回紧凑字符串,重对象(DataFrame、模型结果)存在 `ToolContext` 里跨轮复用**。比如 `load_enso_data` 返回"rows=540,date=1980→2024,best_model=linear_ridge",100 字;实际的 540 行 DataFrame 存在 `ctx.enso`,LLM 不需要看到。`forecast_for_month` 算 lead 分档,12 个月以上直接拒——不是技术做不到,是可预报性物理上限,必须诚实。

## 第 5 页:对话 loop 原理(turn-by-turn)

**标题**:`run_turn`——不是一次跑完,是一轮一轮交还控制权

```python
def run_turn(messages, tools, client, on_step=None):
    while step < max_steps:
        assistant = client.chat(messages, tools.schemas())
        messages.append(assistant)
        if not assistant.tool_calls:
            return TurnResult(messages, assistant.content)  # ← 控制权还给用户!
        for call in assistant.tool_calls:
            result = tools.execute(call.name, call.args)
            on_step(step, name, args, result)  # ← 传给 UI 渲染折叠块
            messages.append({"role":"tool", ...})
```

**讲稿**:这才是对话式 agent 的心脏。旧版本 `run_agent` 是封闭 while 循环:调模型→执行工具→继续→直到模型不调工具,全程不可中断。`run_turn` 的区别**:它不自己建 messages,而是接收外部传入的对话历史**。每轮用户发消息后追加进 history,调一轮直到模型自然停止,然后返回。控制权还给用户——用户随时发下一条消息,追加进 history,再调一次 `run_turn`。三个保护:指数退避重试(429/5xx)、循环检测(连续相同调用 3 次就停)、max_steps 硬上限。

## 第 6 页:状态管理——对话记忆怎么跨轮保持

**标题**:记忆分两块:对话历史 + 重对象

```
session_state
  ├─ messages: [system, ...user1, assistant1, tool_results, user2, ...]
  │   (纯文本,每轮追加)
  │
  └─ ctx: ToolContext(enso=DataFrame, results=dict, figure_paths=[Path])
      (重对象,工具层读写,LLM 不直接感知)
```

**讲稿**:对话记忆分两块。**对话历史**(messages 列表)存文本——system 提示词 + 每轮 user/assistant/tool 消息。这是 LLM 的上下文,也是跨轮"记住刚才聊了什么"的载体。**重对象**(enso DataFrame、模型结果 dict、图文件路径)存在 `ToolContext` 上——这是共享工作记忆,工具层读写,LLM 看不见原始 DataFrame,只通过工具结果字符串了解摘要。`ToolContext` 里的 enso 和 results 第一次被 `load_enso_data` 加载后,后续 `forecast_for_month` 直接读缓存,不用重训。对话历史太长时,`summarize_old_messages` 把旧消息压成摘要,保留最近 6 条。

## 第 7 页:摘要压缩原理

**标题**:对话太长怎么办——自动摘要压缩

```
messages = [system, ...u1...a10]    (token 超阈值)
    ↓
旧消息 [u1..a7] → DeepSeek("摘要要点,不编造数值")
    ↓
新 messages = [原 system, {system: 摘要}, a8, u9, a9, u10, a10]
                 ↑ 保留           ↑ 摘要注入    ↑ 最近 3 轮不动
```

**关键设计**:失败回退——DeepSeek 摘要调用失败→返回原 messages,不阻塞对话。

**讲稿**:这个设计很务实。token 估算是粗粒度的——数字符,不精确,只是触发信号。压缩时把旧消息(除 system 提示词和最近 6 条)发给 DeepSeek,让它"摘要成要点,保留关键预测结果,不要编造数值"。摘要注入成一条 system 消息,和原 system 提示词并存。最近 6 条不动,保证当前对话连贯。最关键:如果摘要调用失败(网络/配额),直接返回原 messages,**宁可不压缩也不崩对话**。

## 第 8 页:LLM 客户端——为什么只留 DeepSeek

**标题**:`LLMClient` 协议 → 单一 `DeepSeekClient`

```python
class LLMClient(Protocol):
    def chat(messages, tools, tool_choice="auto") -> AssistantMessage: ...

class DeepSeekClient:           # OpenAI 兼容,urllib 实现,零额外依赖
    def chat(messages, tools):  # POST /chat/completions
        ...                      # 指数退避重试:429/5xx/网络

# OfflineClient:已删除
# 对话式必须 LLM,离线脚本无法对话
```

**讲稿**:旧 agent 有一个 `OfflineClient`——它不调 LLM,而是按固定 8 步脚本回放(load→plot→precip→tide→report)。对话式 agent 不能离线:每一步都需要 LLM 理解用户意图并选择工具。所以删掉了 OfflineClient,只留 `DeepSeekClient`。它用标准库 `urllib`,不依赖 openai SDK。带了指数退避重试:瞬时错误(429/5xx/网络)自动重试最多 3 次,认证错误(401)立即抛。无 API key 时聊天框直接 disabled,不白跑。

## 第 9 页:用户上传 CSV 的实现

**标题**:`load_user_enso`——让用户自带数据

```
[Streamlit sidebar file_uploader]
    ↓ 上传 my_enso.csv (date+nino34)
[存到 session temp dir]
    ↓ 路径写进 session_state["user_csv_path"]
[用户:"我用上传的数据分析"]
    ↓ LLM 调 load_user_enso(path=path)
[load_enso_csv() 校验列名+行数] → 特征工程 → 模型训练 → 1/3/6 lead 预测
    ↓ ctx.enso ← 用户数据
    ↓ ctx.results ← 新模型结果
[后续 forecast_for_month 自动用用户数据]
```

**讲稿**:上传功能的实现链路:侧栏 file_uploader 存 CSV 到临时目录,把路径写进 session_state。用户在对话里说"加载我的数据",LLM 理解后调 `load_user_enso(path)`。工具内部用 `load_enso_csv` 校验——必须有 date 和 nino34 列,至少 30 行;校验不过返 Error 字符串,不崩。通过后调 `run_forecast_on_enso`(和 `load_enso_data` 共享的核心),训练 Ridge+Random Forest,结果写进 ctx。之后所有预测自动用用户数据。`run_forecast_on_enso` 是从 `run_enso_forecast` 提取出来的:136 行"已有 enso→预测"逻辑抽成独立函数,两种加载方式共享。

## 第 10 页:预测原理——从 Niño3.4 值到 ENSO 阶段

**标题**:预测管道:特征→模型→lead→阶段

```
ENSO 月度序列 (nino34)
    ↓ 特征工程
  [lag_0..lag_12, roll_mean_3, roll_mean_6, month_sin, month_cos]
    ↓ 时间顺序切分(train/test,25% test,不放随机)
    ↓ 三类模型
  Persistence(未来=现在) | Ridge(α=1.0) | RandomForest(120 trees, max_depth=8)
    ↓ 1/3/6 月提前量
  y_{t+h} = model(features_t)
    ↓ RMSE 选最佳模型
    ↓ Niño3.4 值 → classify_phase(±0.5℃)
    ↓ 3 种状态
  El Niño (≥+0.5) / Neutral (-0.5~+0.5) / La Niña (≤-0.5)
```

**讲稿**:预测管道分五步。特征工程:用滞后项(lag_0 到 lag_12)、3 月和 6 月滚动均值、月份周期编码(sin/cos),**只使用当前和历史信息,无未来泄露**。数据按时间顺序切分,不是随机打乱——时间序列必须这样。三类模型:Persistence 是基线(假设未来等于现在,判断 ML 是否有效);Ridge 回归可解释;Random Forest 120 棵树、深度 8。每个 lead 单独评估——1 月、3 月、6 月三个 target。选 RMSE 最低的模型做最终预测。最后用 ±0.5℃ 阈值把连续值变成 El Niño/Neutral/La Niña 三分类。

## 第 11 页:特征工程——防止数据泄露

**标题**:只用历史信息,不用未来信息

```python
def make_enso_supervised_table(df, leads=(1,3,6), max_lag=12):
    for lag in range(max_lag+1):
        data[f"nino34_lag_{lag}"] = data["nino34"].shift(lag)  # ← lag=0 是当前
    # 目标:未来 h 个月的 Niño3.4
    for lead in leads:
        data[f"target_lead_{lead}"] = data["nino34"].shift(-lead)  # ← 负号表示"未来"
    # 时间顺序切分,不是随机
    train, test = temporal_train_test_split(table, test_fraction=0.25)
```

**讲稿**:特征工程最关键的一条:只用当前和历史信息,不允许未来信息泄露。`lag_0` 就是当前值,`lag_1` 是上个月,往后推到 `lag_12`。target 用负号 shift——`shift(-3)` 就是"3 个月后的值"。时间序列切分不用 sklearn 的随机 split,而是按时间点切——前面 75% 训练,后面 25% 测试。这样测试集在训练集的时间之后,符合实际预测场景。

## 第 12 页:关键代码回顾

**标题**:核心模块一览

| 文件 | 行数 | 职责 | 关键函数 |
|---|---|---|---|
| `src/agent/tools.py` | ~580 | 14 个工具 + ToolContext + build_tools | `_forecast_for_month`, `recommend_data_range_dict` |
| `src/agent/run_turn.py` | ~120 | turn-by-turn loop | `run_turn`, `_chat_with_retry` |
| `src/agent/summarizer.py` | ~80 | 对话历史摘要压缩 | `summarize_old_messages`, `estimate_tokens` |
| `src/agent/client.py` | ~200 | DeepSeek API client | `DeepSeekClient.chat`, 指数退避 |
| `src/pipeline/run_enso_forecast.py` | ~240 | ENSO 预测核心 | `run_forecast_on_enso`, `run_enso_forecast` |
| `src/web/app.py` | ~190 | Streamlit chat 粘合 | `main`, `_render_tool_step` |
| `src/features/enso_features.py` | 50 | 特征工程 | `make_enso_supervised_table` |
| `src/models/enso_ml.py` | 65 | Ridge + Random Forest | `build_model_suite`, `fit_models_for_latest_forecast` |
| **总计** | **~2300** | 26 测试,14 工具 | |

**讲稿**:最后过一遍代码规模。总共 2300 行 Python,分四大区:agent(工具+loop+client+摘要)~980 行,科学引擎(data/features/models/analysis/viz)~950 行,pipeline~240 行,web(chat UI)~200 行。26 个测试覆盖工具层、loop、摘要、chat_helpers、预测核心。这不是大项目,但每层职责单一、接口清晰——对话 loop 不懂工具细节,工具不懂 LLM 协议,科学引擎根本不知道 LLM 存在。

## 第 13 页:总结与讨论

**标题**:总结

1. **工具层是关键抽象**——把 ML 模型包装成 LLM 可调用的工具,是"让 AI 做科研"的通路
2. **turn-by-turn 比一次性 loop 更适合对话**——状态外化,用户主导,可插话
3. **设计要诚实地表达不确定性**——lead≥12 不预测,7-11 标低可信度
4. **"重对象在 ctx,文本在 messages"**——这个分离让 LLM 高效消费信息

**讨论点**:
- 如何评估 agent 的"规划"能力?prompt 清单 vs 自主分解
- 对话式 vs 一次性:不同场景选不同形态
- 更多数据源(NOAA/GPCP/ERA5)接入的扩展性

**讲稿**:四点总结。工具层把领域知识封装成 LLM 可调用的接口,这是"让 AI 做科研"的通用模式。turn-by-turn 让用户主导对话,而不是 agent 闷头跑完。lead 可信度分档是诚实的设计——科学上 ENOS 可预报性随 lead 衰减,系统必须如实表达不确定性。最后,"重对象在 ctx,文本在 messages"这个分离,是 agent 能高效运行的核心工程决策。

**页数**:约 13 页,每页 1-2 分钟,总计 15-25 分钟。
