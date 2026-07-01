# ENSO 对话式 Agent:对话驱动的多方法科学预测

> PPT 大纲 + 每页讲稿。主线:**用对话 agent 串起多方法可信预测**——agent 工程与科学方法并重。
>
> 项目:enso-chat/ | ~4500 行 Python | 82 个测试 | 21 个工具 | 双轨预测 + 三层评估

## 叙事主线(讲之前先想清楚)

这个项目不是"套个对话壳的预测脚本",也不是"孤立的科学模型"。核心叙事是:
**对话 agent 让多个预测方法(基线 / 增强 / CNN-LSTM)并存且可对比,并把"可靠性判断"本身交给 LLM 自主调用工具完成。**

评委如果只记一件事,应该是这条:agent 不只是预测,它还能**自己回答"这个预测可不可信"**(调 hindcast 工具看 ACC、对比 Persistence 基准)。

---

## 第 1 页:标题页

**标题**:ENSO 对话式预报 Agent
**副标题**:对话驱动的多方法预测与可靠性自评估
**脚注**:○○○ 课程项目 | 2026-07

**讲稿**:大家好。今天讲一个 ENSO 预报 agent。它的特点不是"能对话",而是"通过对话让多个科学方法协同,并且能自评可靠性"。你说"明年 3 月怎么样,准不准",agent 会自动选方法、预测、然后调工具查这个 lead 的历史技能,告诉你能不能信。我从架构讲到科学方法,再到这套"自评可靠性"是怎么实现的。

## 第 2 页:系统能做什么(能力一览,先建立印象)

**标题**:不只是预测——会自评可靠性的对话 agent

- 对话式预测:自然语言提需求月,agent 自动算 lead、选方法、预测
- **多方法并存**:基线(Ridge/RF) / 增强(+外生指数) / CNN-LSTM(空间场深度学习)
- **可靠性自评估**:agent 主动调 hindcast 工具,用 ACC + Persistence 基准回答"准不准"
- 可信度数据驱动:按 per-lead ACC 分档,<0.3 拒绝、<0.5 低可信
- 数据源注册表:在线拉取 NOAA Niño3.4 / SOI / Niño1+2,agent 自主列源/选源/加载
- 实时空间场:CNN-LSTM 在线拉 OISST+GODAS+NCEP 做真·实时推理
- 对话历史管理:超长自动摘要压缩;CSV 上传自带数据

**讲稿**:先建立印象。这个 agent 有两层能力:预测层有三类方法从简单到复杂并存;评估层让它能回答"预测可不可信"——这是大多数预测系统缺失的一环。数据上既有一维指数在线拉取,也有 CNN-LSTM 需要的空间场实时管道。下面先讲架构,再讲科学方法,最后讲可靠性自评估这条主线。

## 第 3 页:整体架构(四层 + 双轨 + 三评估)

**标题**:四层架构:引擎 → 工具 → 对话 loop,双轨预测 + 三层评估

```
┌──────────────────────────────────────────────┐
│  Streamlit chat UI                           │  交互层
├──────────────────────────────────────────────┤
│  run_turn (turn-by-turn loop)                │  对话调度
│  + summarize_old_messages (历史压缩)         │  (状态外化,每轮交还控制权)
├──────────────────────────────────────────────┤
│  21 个工具 (tools.py)  ToolContext 多槽并存  │  工具层
│  results / enhanced_results / cnn_forecasts  │  (重对象在 ctx,文本在 messages)
├──────────────────────────────────────────────┤
│  双轨预测                三层评估             │  科学层
│  ├ baseline (Ridge/RF)   ├ SODA hindcast     │
│  ├ enhanced (+SOI/N12)   ├ Realtime hindcast │
│  └ CNN-LSTM (空间场)     └ Persistence 基准  │
└──────────────────────────────────────────────┘
```

**讲稿**:整系统四层。底下科学层是关键:预测分双轨——一维指数轨(基线+增强,实时主力)和空间场轨(CNN-LSTM,方法上限);评估分三层——SODA 域 hindcast(训练域,方法上限)、Realtime 域 hindcast(推理域,真实跨域效果)、Persistence 基准(两域都有,技能锚)。工具层把双轨+评估都包成工具,ToolContext 用多槽让三方法结果并存不互相覆盖。对话层是 turn-by-turn,每轮交还控制权。这张图是全局地图,后面每页展开一块。

## 第 4 页:对话 loop——turn-by-turn 状态外化(agent 核心)

**标题**:`run_turn`——不自己持有状态,借用调用方的对话历史

```python
def run_turn(messages, tools, client, on_step=None):
    while step < max_steps:
        assistant = client.chat(messages, tools.schemas())
        messages.append(assistant)          # ← 原地追加外部列表
        if not assistant.tool_calls:
            return TurnResult(messages, assistant.content)  # ← 控制权还给用户
        for call in assistant.tool_calls:
            result = tools.execute(call.name, call.args)
            on_step(...)                    # ← 回调给 UI 渲染折叠块
            messages.append({"role":"tool", ...})
```

**三重保护**:指数退避重试(429/5xx/网络) · 循环检测(连续相同签名 3 次早停) · max_steps 硬上限(15)。

**讲稿**:对话式 agent 的心脏。和一次性 `run_agent` 的根本区别:它不自己建 messages,而是接收外部传入的对话历史,原地追加。这样 Streamlit 的 `session_state.messages` 就是天然持久层,用户下一条消息追加进来再调一次 `run_turn`,上下文连续。三个保护保证鲁棒:瞬时错误退避重试、模型卡死循环早停、单轮工具调用上限防失控。`on_step` 回调把每次工具调用传给 UI 渲染折叠块——loop 不耦合 Streamlit,纯函数可测。

## 第 5 页:工具层——"重对象在 ctx,文本在 messages"

**标题**:21 个工具,共享 ToolContext 多槽状态

| 类别 | 工具 | 关键设计 |
|---|---|---|
| 数据 | `load_enso_data` `load_user_enso` `load_index` `list_data_sources` `diagnose_local_data` | 幂等缓存;数据源注册表 |
| 预测 | `forecast_for_month` `forecast_latest` `forecast_enhanced` `forecast_cnn_lstm` | 双轨并存,mode 区分 soda_tail/realtime |
| 评估 | `report_hindcast_skill` `report_realtime_skill` `recommend_data_range` `compare_methods` | **可靠性自评估工具集** |
| 分析画图 | `classify_phase` `analyze_precipitation` `run_tide_prediction` + 4 画图 | matplotlib→PNG 内联展示 |

**核心原则**:工具返回**紧凑字符串摘要**(路径+关键数字),重对象(DataFrame/results/权重)存在 `ToolContext` 多槽——`results`(基线)/`enhanced_results`(增强)/`cnn_forecasts`(CNN)并存,不互相覆盖。

**讲稿**:工具层两个要点。第一,每个工具返回字符串而非 DataFrame——540 行序列化进 messages 会爆上下文,所以重对象存 ctx,LLM 只看摘要。第二,多方法结果分槽并存:基线、增强、CNN 各占一个槽,agent 同会话调三方法做对比时不会互相覆盖。表里高亮的是评估类工具——这是项目特色,agent 不只预测,还能调 `report_hindcast_skill` 查历史技能、`compare_methods` 并排对比,实现"自评可靠性"。

## 第 6 页:科学方法(1)——基线轨与防数据泄露

**标题**:基线轨:Persistence + Ridge + Random Forest,严格无泄露

```
Niño3.4 月度序列
  ↓ 特征:lag_0..12 + 滚动均值3/6 + month sin/cos   (只用当前和历史)
  ↓ 目标:shift(-lead) 未来值                        (负号=未来)
  ↓ 时间顺序切分 train/test (75/25,不随机打乱)
  ↓ 三模型:Persistence(基线) / Ridge(α=1.0) / RF(120树,depth8)
  ↓ RMSE 选最优 + ACC 进结果(供数据驱动分档)
```

**防泄露三道防线**:特征只用 lag≥0(历史) · 目标用 shift(-lead)(未来) · 时间序切分(非随机)。

**讲稿**:基线轨是实时预测的底线。特征工程只允许当前和历史信息进入——lag_0 是当前值,lag_12 是一年前,绝不用 shift(-k) 当特征(那是未来)。目标用负号 shift,表示要预测的未来值。切分按时间顺序,绝不用 sklearn 随机 split——那会让测试集时间点混进训练集之前,形成时间泄露。三模型里 Persistence 是基线中的基线(假设未来=现在),用来判断 ML 是否真有效——如果 ML 不如 Persistence,说明特征或模型有问题。每个 lead 的 RMSE 和 ACC 都进结果,ACC 供后面的数据驱动分档用。

## 第 7 页:科学方法(2)——增强轨:外生气候指数

**标题**:增强轨:加 SOI + Niño1+2 突破单变量瓶颈

**问题**:基线只用 Niño3.4 自身滞后,撞春季预测障碍(SPB)——能跨春的前兆信号在**大气端(SOI)和东太平洋海洋端(Niño1+2)**。

**方案**:
- 数据源注册表(`source_registry.py`):NOAA/PSL 三源在线拉取,免注册 ASCII
- `enso_features` 加 `exog_cols`:SOI/Niño1+2 各加 13 个 lag 特征
- `forecast_enhanced`:Niño3.4 + 外生指数 → Ridge/RF
- **lead 可信度数据驱动**:读 per-lead ACC,<0.3 拒绝、<0.5 低可信(替代硬编码 7/12)

**讲稿**:基线轨的天花板在物理——ENSO 跨春预测靠的不是 Niño3.4 自己,而是次表层和大气前兆。增强轨加 SOI(南方涛动指数,大气端)和 Niño1+2(东太平洋上涌区,海洋端)两个外生指数。这两个都是 NOAA/PSL 在线可拉的一维月值,套现有解析器,实时可用。关键改进:lead 可信度不再硬编码"7-11 月低可信",而是读这个 lead 在测试集上的 ACC——ACC 跌破 0.5 才标低可信、跌破 0.3 才拒绝。这让分档有数据支撑,答辩能讲清"为什么这个 lead 不可信"。

## 第 8 页:科学方法(3)——CNN-LSTM 空间场轨

**标题**:CNN-LSTM:sst/t300/ua/va 空间场,对标 Ham et al. 2019 (Nature)

```
输入:12 月 × (24纬×72经) × 4 通道  (sst/t300/ua/va)
  ↓ CNN 提空间特征(每时间步)
  ↓ LSTM 建时序(2层)
  ↓ FC 输出 24 lead Niño3.4
```

- **训练**:SODA 再分析,留缓冲三划分(训0-70/验70-82/缓冲82-85/测85-99),防时序渗透
- **标准化**:只用训练集统计量;NaN→0;BatchNorm+Dropout(0.7) 抗小样本过拟合
- **离线训练 / 在线推理分离**:权重 49M 内置,torch 懒加载,Streamlit 无依赖启动
- **复用参考 notebook 架构**(CNN-LSTM),SODA-only 训练(无 CMIP 数据)

**讲稿**:CNN-LSTM 是方法上限。参考 Ham et al. 2019 Nature 论文和提供的参考 notebook,用四个空间场通道:CNN 提取每个时间步的空间特征,LSTM 建时序,一次输出 24 个 lead。训练在 SODA 再分析上,关键是划分留了缓冲区——测试集窗口起点在 85 年后,和训练尾完全隔开,防止时序渗透虚高 ACC。工程上离线训练、在线只做 CPU 前向推理,权重内置仓库,Streamlit 启动不需要 torch。这是和基线/增强轨并存的第三种方法,不是替代。

## 第 9 页:可靠性评估(1)——Hindcast 协议与 SODA 域技能

**标题**:怎么判断预测准不准——hindcast + Persistence 基准(对标 Nature 论文)

**核心**:预测可靠性不是"感觉",是**该 lead 在历史测试集上的 ACC**(异常相关系数),对标 Ham et al. 2019 口径。

| lead | CNN-ACC | Persistence | gap | 判断 |
|---|---|---|---|---|
| 1-3 | 0.77/0.67/0.59 | 0.91/0.77/0.61 | **负** | Persistence 略胜(ENSO 短期自相关) |
| 4-13 | 0.53→0.48 | 0.45→-0.15 | **正且扩大** | CNN 有真技能,可靠窗口起点 |
| 10 (SPB谷) | 0.31 | **-0.10** | +0.40 | Persistence 已失效,CNN 仍正技能 |
| 14-24 | ~0.52 | -0.16→0.58 | 正→负 | CNN 稳在 0.5+,对标论文 lead17>0.5 |

**结论**:可靠窗口 lead 4-23;CNN 的价值在 Persistence 失效的中长 lead;长 lead 量级对齐 Nature 论文。

**讲稿**:这页回答"准不准"。可靠性不能靠感觉,要用 hindcast——在历史测试集上算每个 lead 的 ACC,这是 Ham et al. Nature 论文用的口径。关键要对比 Persistence 基准:短期(lead1-3)Persistence 反而比 CNN 强,因为 ENSO 短期自相关太强,"明天=今天"几乎不可战胜,这正常;但 lead4 起 CNN 开始碾压,到 lead10 春季预测障碍低谷时 Persistence 已经负相关、CNN 仍是正技能——CNN 的存在价值就在这里。长 lead CNN 稳在 0.5+,和论文"lead17 仍>0.5"同量级。这张表是答辩最有力的证据:可靠窗口、SPB 可见、对标论文,全有了。

## 第 10 页:可靠性评估(2)——Realtime 域与跨域诚实

**标题**:训练域≠推理域——Realtime hindcast 才是真效果

**问题**:CNN 在 SODA 训练,推理却喂 OISST+GODAS+NCEP。**SODA 域 ACC 不能直接套到 realtime**。

**方案**:用 realtime 源历史段做独立 hindcast,对比真实 Niño3.4,算跨域 ACC。
- 泄漏无关气候态:气候态期(2005-2015)与评估期(2020-2021)不重叠
- 异常化对齐:realtime 源减自身气候态转异常,匹配 SODA 的异常分布
- 同口径:all-season ACC + Persistence 基准

**结果**(n=7 窗口,短 lead 可信):
- lead1=0.86, lead2=0.75 — **跨域损失不严重,realtime 短期可靠**
- 中长 lead 因 n=7 样本小,标注"待扩样本"

**诚实底线**:realtime 结果永远标注"cross-domain,精度低于 SODA hindcast",不混淆两域 ACC。

**讲稿**:这页是科学诚实的核心。SODA 域 ACC 0.77 不能直接说"realtime 也 0.77"——因为训练和推理用了不同数据源,存在跨域偏移。所以单独建 realtime 域 hindcast:用 OISST/GODAS/NCEP 历史段跑 CNN,对比真实 Niño3.4。结果短 lead 跨域效果不错(0.86/0.75),说明异常化对齐有效;中长 lead 因为样本只有 7 个窗口,结论不稳,诚实标注"待扩样本"。最关键的是工具返回永远带"cross-domain"标注,agent 不会拿 SODA 的 ACC 冒充 realtime——这是大多数预测系统不敢做的诚实。

## 第 11 页:实时数据管道——从"拿不到最新数据"到真·实时

**标题**:CNN-LSTM 实时化:OISST+GODAS+NCEP,免注册 OPeNDAP

**难点**:CNN 推理需 12 月 × 4 通道空间场,SODA 末端是历史固定点,不是"现在"。

**方案**:
| 通道 | 源 | 滞后 | 工程坑(已解决) |
|---|---|---|---|
| sst | NCEI OISST 日值(月中近似) | ~1-2周 | 中文路径读 nc→临时ASCII拷贝;zlev 维 squeeze |
| t300 | PSL GODAS pottmp(303m层) | ~1月 | OPeNDAP interp跨层返回0→sel nearest |
| ua/va | PSL NCEP/NCAR R1 850hPa | ~5月 | 整文件437MB断连→OPeNDAP切片;3D interp返回0→逐片 |

**窗口截止**:风场5月滞后是瓶颈,窗口统一截止到风场最新月,诚实标注不伪造填充。

**讲稿**:这页解决"拿不到最新数据"。CNN-LSTM 原来只能用 SODA 末端,没法做真·实时。实时管道在线拉四个源,全是免注册 OPeNDAP。过程修了一堆环境坑:NCEP 整文件 437MB 下载断连改 OPeNDAP 切片;OPeNDAP 对 3D 数据 interp 返回 0 改逐时间片;netCDF4 读不了中文路径用临时 ASCII 拷贝。风场滞后 5 个月是瓶颈,所以窗口截止到风场最新月,sst/t300 更新的部分被截掉——诚实标注,不拿旧风场填充假装实时。这页的工程含量很高,但答辩重点是"真要能用"这个目标怎么落地。

## 第 12 页:对话驱动的科学决策(主线收束)

**标题**:agent 自主编排多方法 + 自评可靠性

**真实对话示例**(用户:"明年3月怎样?对比三方法,准不准"):

```
agent 自主 7 步编排:
  1. load_enso_data(auto→真实NOAA)     4. recommend_data_range(2027-03→lead11低可信)
  2. load_index(soi)  3. load_index(nino12)   (在线拉外生指数)
  5. forecast_enhanced(2027-03)  → ACC=0.38 自动标低可信
  6. forecast_cnn_lstm(realtime, lead11)  → 跨域标注
  7. forecast_latest(lead1)  → 主动补短期 La Niña
  → 生成三方法对比表 + 可靠性解读
```

**讲稿**:这页是整条主线的收束。agent 不是被动的"问什么答什么",它收到"对比三方法+准不准"后,自主编排了 7 步:加载数据、拉外生指数、算 lead、跑增强预测(并用 ACC 解释低可信)、跑 CNN 实时预测、主动补一个短期预测、最后生成对比表。最关键的是第 5 步——agent 拿到 ACC=0.38 后,**自发**告诉用户"这个 lead 可信度低,建议6月内再关注"。这就是"对话驱动科学决策"的完整闭环:多方法 + 自评可靠性 + 诚实建议,全部 LLM 自主调度。评委如果问"agent 比 scripts 强在哪",这页就是答案。

## 第 13 页:关键代码与规模

**标题**:核心模块一览

| 区 | 文件 | 职责 |
|---|---|---|
| agent | `run_turn.py` `client.py` `summarizer.py` `tools.py` | 对话 loop + 21 工具 + DeepSeek client |
| 科学引擎 | `features/` `models/` `analysis/` `visualization/` `pipeline/` | Ridge/RF + CNN-LSTM + 特征 + 画图 |
| 数据 | `noaa_enso.py` `source_registry.py` `realtime_fetch.py` `climatology.py` | NOAA拉取 + 注册表 + 实时管道 + 气候态 |
| 评估 | `hindcast.py` `realtime_hindcast.py` `evaluation.py` | SODA/Realtime hindcast + ACC |
| web | `app.py` `chat_helpers.py` | Streamlit chat 粘合 |
| 离线脚本 | `train_cnn_lstm.py` `build_climatology.py` `run_hindcast.py` `run_realtime_hindcast.py` | 训练/气候态/评估(不进 Streamlit) |

**总计**:~4500 行 Python,82 测试,21 工具,4 个离线脚本,49M CNN 权重内置。

**讲稿**:代码规模。四大区:agent(~1100行)、科学引擎(~1300行)、数据与评估(~900行,含实时管道)、web(~250行)。82 个测试覆盖工具层、loop、摘要、hindcast、实时管道,零网络依赖隔离测试。四个离线脚本负责训练和评估,不进 Streamlit 进程——这是"离线训练/在线推理分离"的体现。CNN 权重 49M 内置仓库,推理 CPU 即可。

## 第 14 页:总结与讨论

**标题**:总结

1. **对话 agent 串起多方法科学预测**——基线/增强/CNN-LSTM 并存且可对比,不是套壳
2. **可靠性自评估是核心特色**——agent 调 hindcast 工具用 ACC+Persistence 回答"准不准"
3. **科学诚实贯穿全系统**——lead 按 ACC 数据驱动分档;跨域不混淆 ACC;SPB 可见
4. **工程务实**——OPeNDAP 切片、离线训练/在线推理、多槽状态并存、零网络测试隔离

**讨论点**:
- agent 的"规划"能力边界:prompt 引导 vs 自主分解(本项目 7 步自主编排)
- realtime hindcast 样本扩展(2018-2021,~30窗口)可让中长 lead 结论更稳
- 标准气候态(1991-2020)替换迷你版,提升科学严谨度
- agent 路由优化:两个 hindcast 工具描述区分,避免误调

**讲稿**:四点总结。第一,这个 agent 真正的价值是让多方法协同并对比,不是给预测套个对话壳。第二,可靠性自评估是特色——大多数预测系统只给数字不给"能不能信",这个 agent 能。第三,科学诚实贯穿始终:数据驱动分档、跨域不混淆、SPB 可见。第四,工程上务实解决真问题:OPeNDAP 切片绕开大文件、离线在线分离、多槽状态、测试隔离。讨论点留几个诚实的不完美:realtime 样本待扩、气候态待标准化、agent 路由待优化——这些是"做满"的项,不影响当前答辩完整性。

**页数**:约 14 页,每页 1.5-2 分钟,总计 20-25 分钟。
