# ENSO 对话式 Agent

Streamlit chat 全对话式 ENSO 预报 agent。用户自由对话，DeepSeek 驱动 ENSO 工具
（预测 / 诊断 / 画图 / 分析），每轮返回控制权，可随时插话。

## 设置

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
export DEEPSEEK_API_KEY=sk-your-key   # 或在页面侧栏填入
```

## 运行

```bash
streamlit run src/web/app.py
```

## 能力

- 对话式预测：自然语言提需求月（"明年3月"等），agent 自动算 lead 并按可信度分档。
- lead 可信度：1-6 月正常，7-11 月低可信度，≥12 个月拒绝预测。
- 工具：加载数据、forecast_for_month、诊断、推荐数据范围、画图、降水/潮汐分析。
- 历史超长自动摘要压缩，保持上下文。

## 测试

```bash
python -m pytest
```

## 与 `agent/` 项目的关系

独立项目，复制了 `agent/src/` 的科学引擎并精简（去掉报告生成、一次性流水线工具、
OfflineClient）。loop 新写 `run_turn`（turn-by-turn），不碰一次性 `run_agent`。

## 架构要点

- **状态**：对话历史存在 Streamlit `session_state`（纯内存，关页面即丢）。
- **loop**：`run_turn` 接收外部 messages 列表，跑到模型不调工具就返回，把控制权交回用户。
- **工具展示**：每次工具调用在对话流插入折叠块，显示调用参数与结果。
- **历史压缩**：估算 token 超阈值时，把旧消息发给 DeepSeek 压成摘要，注入为 system 消息，保留最近 6 条。失败时回退原历史。
- **错误处理**：任一轮失败都能让用户继续发新消息（失败信息进气泡，历史仍可用）。
