"""Self-description & source-reading capability for the agent.

Two read-only helpers wired up as agent tools so the agent can explain **how
each component of itself is implemented** and **read its own source code** on
demand — both grounded in the real repo (no fabrication).

- :func:`explain_component` — returns a structured summary (responsibility, key
  functions/classes, dependencies, source file) for a registered component,
  drawn from a hand-authored ``COMPONENTS`` table. The table is short prose +
  pointers, not verbatim code, so it stays token-cheap and lets the LLM produce
  the actual explanation rather than dumping a whole file. When a function-level
  detail is needed, the agent follows up with :func:`read_source`.

- :func:`read_source` — reads a real file from the repo with line numbers,
  optionally locating a single ``symbol`` (function/class def line) or a line
  range. Output is hard-capped (``MAX_SOURCE_CHARS``) so a 1500-line module
  never floods the context. Paths must be under the project root (sandboxed).

Both are pure, side-effect-free, and trivially unit-testable without an LLM.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# Project root: this file is src/agent/code_guide.py → parents[2] = enso-chat/.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Hard cap on returned source text so a request for a huge file never floods the
# agent's context window. When exceeded, the tail is dropped with a marker.
MAX_SOURCE_CHARS = 6000


from dataclasses import dataclass


@dataclass(frozen=True)
class ComponentInfo:
    """One registered component of the project."""

    key: str  # short id, e.g. "agent.run_turn"
    path: str  # repo-relative source path
    layer: str  # agent / data / features / models / pipeline / analysis / viz / web / reports / scripts
    responsibility: str  # one-paragraph "what it does / why"
    key_symbols: list[str]  # functions/classes/tools the agent may want to read
    dependencies: list[str]  # notable imports / sibling modules
    related_tools: list[str]  # agent tools this component backs (if any)


# Hand-authored table — every entry is grounded in the real source I read.
# When a user asks "X 是怎么实现的", explain_component(key) returns this; the
# agent follows up with read_source(path, symbol=...) if function-level detail
# is requested. Keep prose short; the table is a map, not a textbook.
COMPONENTS: dict[str, ComponentInfo] = {
    # --- agent layer ---
    "agent.run_turn": ComponentInfo(
        key="agent.run_turn",
        path="src/agent/run_turn.py",
        layer="agent",
        responsibility=(
            "turn-by-turn 对话循环（区别于一次性 run_agent）。接收外部 messages 列表，跑到模型不再调工具即返回、"
            "把控制权交回调用方，状态外化在调用方持有的列表上原地修改。含工具循环检测（连续相同 (tool,args) "
            "≥loop_limit 次提前终止）与 max_steps 硬上限，以及对 client.chat 的指数退避重试。"
        ),
        key_symbols=["run_turn", "TurnResult", "_chat_with_retry", "_freeze"],
        dependencies=["src.agent.client", "src.agent.tools", "src.config"],
        related_tools=["(all tools — this is the loop that drives them)"],
    ),
    "agent.client": ComponentInfo(
        key="agent.client",
        path="src/agent/client.py",
        layer="agent",
        responsibility=(
            "DeepSeek LLM 客户端，OpenAI 兼容 chat-completions over urllib（无额外依赖）。"
            "tool_calls 参数在 client 内即时解析为 dict，使上游 loop 与具体厂商无关。"
            "DeepSeekError 带 retryable 标志：429/5xx/网络错误可重试，401/403/400 不可重试。"
            "deepseek-reasoner 不支持 function calling，会被 _resolve_deepseek_config 显式拒绝。"
        ),
        key_symbols=["DeepSeekClient", "AssistantMessage", "ToolCall", "DeepSeekError", "LLMClient", "_resolve_deepseek_config"],
        dependencies=["urllib", "src.config"],
        related_tools=[],
    ),
    "agent.glm_client": ComponentInfo(
        key="agent.glm_client",
        path="src/agent/glm_client.py",
        layer="agent",
        responsibility=(
            "智谱 GLM 客户端，复用 DeepSeekClient 的请求/解析机制（同 OpenAI 兼容 schema: tools/tool_calls/tool_call_id），"
            "因此 agent loop 无需改动即可切换。从 GLM_API_KEY 环境变量取 key。"
        ),
        key_symbols=["GLMClient", "_resolve_glm_config"],
        dependencies=["src.agent.client", "src.config"],
        related_tools=[],
    ),
    "agent.summarizer": ComponentInfo(
        key="agent.summarizer",
        path="src/agent/summarizer.py",
        layer="agent",
        responsibility=(
            "历史压缩：估算 token 超过 20000 字符阈值时，把旧消息压成摘要注入为新 system 消息、保留最近 6 条。"
            "估算故意粗（按字符数），仅驱动触发。任何失败都回退原历史——绝不因摘要失败阻断对话。"
        ),
        key_symbols=["summarize_old_messages", "should_summarize", "estimate_tokens"],
        dependencies=["src.agent.client"],
        related_tools=["(used by app.py between turns)"],
    ),
    "agent.tools": ComponentInfo(
        key="agent.tools",
        path="src/agent/tools.py",
        layer="agent",
        responsibility=(
            "工具层：把 src/ 既有科学函数包装为 agent 可调工具（name+JSON-Schema 参数+callable）。"
            "ToolContext 持有跨工具的共享可变状态（ENSO 序列、results、enhanced_results、cnn_forecasts、figure_paths、report_path）。"
            "覆盖三轨预测、画图、数据来源、回算技巧、报告生成/润色等二十余件工具。"
            "ToolRegistry 仅转发调用并把异常以字符串形式回给 LLM（崩不了 loop）。"
        ),
        key_symbols=["ToolContext", "Tool", "ToolRegistry", "build_tools", "_forecast_for_month",
                     "_forecast_cnn_lstm", "_forecast_enhanced", "_compare_methods",
                     "_report_hindcast_skill", "_report_realtime_skill", "_write_forecast_report",
                     "_read_report", "_accept_report_polish"],
        dependencies=["src.pipeline.run_enso_forecast", "src.models", "src.visualization.plots", "src.data.source_registry"],
        related_tools=["load_enso_data", "forecast_for_month", "forecast_cnn_lstm", "forecast_enhanced",
                       "compare_methods", "report_hindcast_skill", "report_realtime_skill",
                       "write_forecast_report", "read_report", "accept_report_polish"],
    ),
    # --- data layer ---
    "data.loaders": ComponentInfo(
        key="data.loaders",
        path="src/data/loaders.py",
        layer="data",
        responsibility="三个 CSV 加载器（ENSO / 降水 / 潮汐），各自校验必需列并按时间排序。ENSO 必含 date+nino34。",
        key_symbols=["load_enso_csv", "load_precipitation_csv", "load_tide_csv"],
        dependencies=["pandas"],
        related_tools=["load_user_enso"],
    ),
    "data.noaa_enso": ComponentInfo(
        key="data.noaa_enso",
        path="src/data/noaa_enso.py",
        layer="data",
        responsibility=(
            "NOAA/PSL Niño3.4 ASCII 月值时序的下载+解析（YYYY v1..v12，-99.99 缺测）。"
            "load_or_download_noaa_enso 带原始+已处理缓存，支持强制刷新。"
        ),
        key_symbols=["load_or_download_noaa_enso", "parse_noaa_nino34_table", "download_noaa_enso_text", "NoaaEnsoDownloadError"],
        dependencies=["urllib", "src.config"],
        related_tools=["load_enso_data"],
    ),
    "data.source_registry": ComponentInfo(
        key="data.source_registry",
        path="src/data/source_registry.py",
        layer="data",
        responsibility=(
            "气候指数数据发现层：登记 Niño3.4 / SOI / Niño1+2 三个 NOAA/PSL 同格式月值时序（每个 DataSource 含 name/url/coverage/value_col）。"
            "parse_year_month_table 泛化解析；load_index 带 raw+processed 缓存。加源只需在此登记一行不改工具。"
        ),
        key_symbols=["REGISTRY", "DataSource", "load_index", "parse_year_month_table", "list_sources", "IndexLoadError"],
        dependencies=["urllib", "src.config"],
        related_tools=["list_data_sources", "load_index", "forecast_enhanced"],
    ),
    "data.realtime_fetch": ComponentInfo(
        key="data.realtime_fetch",
        path="src/data/realtime_fetch.py",
        layer="data",
        responsibility=(
            "实时空间场抓取：从 NCEI OISST / PSL GODAS pottmp / PSL NCEP R1 uwnd,vwnd 拉月值场，"
            "重采样到 SODA 训练网格（24×72, 5°），用预置气候态反距平化，拼成 12 月输入窗。"
            "风道滞后约 5 个月最差，窗截到风道最新月；单道失败零填并标注 degraded，不整轮崩。"
        ),
        key_symbols=["fetch_realtime_window", "ChannelResult", "RealtimeFetchError"],
        dependencies=["urllib", "xarray", "src.data.climatology", "src.config"],
        related_tools=["forecast_cnn_lstm(mode='realtime')"],
    ),
    "data.climatology": ComponentInfo(
        key="data.climatology",
        path="src/data/climatology.py",
        layer="data",
        responsibility=(
            "实时轨的反距平化必需：CNN-LSTM 训练于 SODA 距平（均值≈0），实时源是绝对值，不做反距平化会造成致命域偏移。"
            "compute_monthly_climatology 按 1991–2020（WMO 30 年气候态）逐月求均，存 .npz；anomalize 用其减得距平。"
        ),
        key_symbols=["compute_monthly_climatology", "anomalize", "load_climatology", "CLIMATOLOGY_YEARS"],
        dependencies=["numpy", "pandas"],
        related_tools=["forecast_cnn_lstm(mode='realtime')"],
    ),
    "data.sample_generator": ComponentInfo(
        key="data.sample_generator",
        path="src/data/sample_generator.py",
        layer="data",
        responsibility=(
            "无网/离线兜底：生成合成的 ENSO / 降水 / 潮汐 样本数据，保证全程离线可跑。"
            "NOAA 不可达时 run_enso_forecast 以 sample 兜底并标注 fallback_reason。"
        ),
        key_symbols=["generate_sample_enso", "generate_sample_precipitation", "generate_sample_tide", "write_sample_datasets"],
        dependencies=["numpy", "pandas"],
        related_tools=["load_enso_data(data_source='sample')"],
    ),
    # --- features ---
    "features.enso_features": ComponentInfo(
        key="features.enso_features",
        path="src/features/enso_features.py",
        layer="features",
        responsibility=(
            "监督表构造：Niño3.4 的 0..12 月滞后(13) + 3/6 月滚动均值(2) + month_sin/cos(2)；"
            "exog_cols 给定时为每个外源指数追加 0..12 月滞后(每指数 13)。"
            "target_lead_{lead}=nino34.shift(-lead)。dropna 去首尾缺失。"
        ),
        key_symbols=["make_enso_supervised_table"],
        dependencies=["numpy", "pandas"],
        related_tools=["load_enso_data", "forecast_enhanced", "forecast_for_month"],
    ),
    # --- models ---
    "models.enso_ml": ComponentInfo(
        key="models.enso_ml",
        path="src/models/enso_ml.py",
        layer="models",
        responsibility=(
            "基础/增强轨模型套件与训练：linear_ridge=Pipeline(StandardScaler→Ridge(alpha=1))；"
            "random_forest=RandomForestRegressor(n_estimators=120,max_depth=8,min_samples_leaf=3,random_state=42)。"
            "train_and_predict_for_lead 每 lead 单独 fit 并预测测试集；fit_models_for_latest_forecast 在全表 fit 后用末行推断。"
        ),
        key_symbols=["build_model_suite", "train_and_predict_for_lead", "fit_models_for_latest_forecast"],
        dependencies=["scikit-learn"],
        related_tools=["load_enso_data", "forecast_enhanced", "compare_methods"],
    ),
    "models.baseline": ComponentInfo(
        key="models.baseline",
        path="src/models/baseline.py",
        layer="models",
        responsibility="Persistence 基线：预报=最近观测月（nino34_lag_0），不训练，作所有预测的对照。",
        key_symbols=["persistence_predict"],
        dependencies=["numpy", "pandas"],
        related_tools=["report_hindcast_skill", "report_realtime_skill"],
    ),
    "models.cnn_lstm": ComponentInfo(
        key="models.cnn_lstm",
        path="src/models/cnn_lstm.py",
        layer="models",
        responsibility=(
            "CNN-LSTM 空间场模型：输入 (12,24,72,4)=12 月 × 24×72 × sst/t300/ua/va；"
            "结构 Conv2d(4→16,k7,s2)→BN→ReLU→Conv(16→16,k3)→Dropout0.7→BN→ReLU→AvgPool→Flatten1728"
            "→LSTM(1728→1024)→LSTM(1024→256)→Dropout0.7→Linear(256→24) 一次出 24 lead。"
            "训练 Adam lr=1e-3 wd=0.001 batch=8 epochs=80 patience=10 ReduceLROnPlateau MSE seed=42；"
            "逐通道用训练集统计量标准化(存 checkpoint)。SODA 划分 train0-71/val70-82/buffer82-84(防泄漏)/test85-100 块。"
        ),
        key_symbols=["_build_model", "train_cnn_lstm", "predict_cnn_lstm", "predict_cnn_lstm_realtime",
                     "make_cnn_lstm_dataset", "load_soda_tail_window", "SPLIT_MONTH_RANGES", "CHANNELS"],
        dependencies=["torch", "xarray", "numpy"],
        related_tools=["forecast_cnn_lstm"],
    ),
    "models.hindcast": ComponentInfo(
        key="models.hindcast",
        path="src/models/hindcast.py",
        layer="models",
        responsibility=(
            "CNN-LSTM 回算技巧评估：在 SODA 测试窗跑训练好的网络，比对 Persistence 基线，报每 lead 的 all-season ACC 与 skill_gap。"
            "指标遵循 Ham et al. 2019。SODA year/month 是匿名竞赛索引，故 per-target-month 仅按块相位标注、不可对真季节。"
        ),
        key_symbols=["run_hindcast", "HindcastResult", "save_hindcast_report", "hindcast_report_text"],
        dependencies=["torch", "numpy", "src.models.cnn_lstm", "src.models.evaluation"],
        related_tools=["report_hindcast_skill"],
    ),
    "models.realtime_hindcast": ComponentInfo(
        key="models.realtime_hindcast",
        path="src/models/realtime_hindcast.py",
        layer="models",
        responsibility=(
            "CNN-LSTM 跨域回算：在 realtime 域(OISST/GODAS/NCEP)评估，给出唯一能评判 realtime 预测的 ACC。"
            "SODA 回算 ACC 不可跨域套用——这正是此模块存在的理由。"
        ),
        key_symbols=["run_realtime_hindcast"],
        dependencies=["torch", "numpy", "src.models.hindcast", "src.data.realtime_fetch"],
        related_tools=["report_realtime_skill"],
    ),
    "models.evaluation": ComponentInfo(
        key="models.evaluation",
        path="src/models/evaluation.py",
        layer="models",
        responsibility=(
            "评估指标：temporal_train_test_split 按时间序末留测试集(防穿越)；"
            "calculate_regression_metrics 给 RMSE/MAE/corr；"
            "calculate_acc = 距平相关系数(各自减均值后的 Pearson 相关)，ENSO 技巧标准度量；"
            "per_lead_metrics 对多 lead 列逐列算 RMSE/MAE/ACC。"
        ),
        key_symbols=["temporal_train_test_split", "calculate_regression_metrics", "calculate_acc", "per_lead_metrics"],
        dependencies=["numpy", "scikit-learn"],
        related_tools=["load_enso_data", "report_hindcast_skill"],
    ),
    "models.tide_model": ComponentInfo(
        key="models.tide_model",
        path="src/models/tide_model.py",
        layer="models",
        responsibility=(
            "潮汐演示预测：用 sin/cos(12.42h M2 半日潮 + 24h)谐波特征 + Ridge(alpha=0.1)，75/25 时序划分，"
            "给 RMSE/MAE/corr 与观测vs预测图。演示用，非实时。"
        ),
        key_symbols=["run_tide_demo_prediction", "TidePredictionResult"],
        dependencies=["scikit-learn", "matplotlib", "src.models.evaluation"],
        related_tools=["run_tide_prediction"],
    ),
    # --- analysis ---
    "analysis.enso_phase": ComponentInfo(
        key="analysis.enso_phase",
        path="src/analysis/enso_phase.py",
        layer="analysis",
        responsibility="ENSO 相位分类：>=0.5 厄尔尼诺、<=-0.5 拉尼娜、其余中性（±0.5 阈值）。add_enso_phase 给 DataFrame 加相位列。",
        key_symbols=["classify_enso_phase", "add_enso_phase"],
        dependencies=["pandas"],
        related_tools=["classify_phase", "forecast_for_month"],
    ),
    "analysis.precipitation_analysis": ComponentInfo(
        key="analysis.precipitation_analysis",
        path="src/analysis/precipitation_analysis.py",
        layer="analysis",
        responsibility="按 ENSO 相位分组的降水距平统计(mean/std/count)与箱线图；ENSO 与降水按日期内连接。",
        key_symbols=["analyze_precipitation_by_enso_phase", "PrecipitationAnalysisResult"],
        dependencies=["matplotlib", "pandas", "src.analysis.enso_phase"],
        related_tools=["analyze_precipitation"],
    ),
    # --- pipeline ---
    "pipeline.run_enso_forecast": ComponentInfo(
        key="pipeline.run_enso_forecast",
        path="src/pipeline/run_enso_forecast.py",
        layer="pipeline",
        responsibility=(
            "端到端建模流水：_resolve_enso_data(sample/noaa/auto 带回退) → run_forecast_on_enso"
            "(做监督表、训练 Persistence+Ridge+RF [lead1/3/6]、评估含 ACC、写 results JSON + predictions CSV)。"
            "支持 exog_cols(增强轨)。load_user_enso 复用 run_forecast_on_enso。"
        ),
        key_symbols=["run_enso_forecast", "run_forecast_on_enso", "_resolve_enso_data", "EnsoForecastOutput"],
        dependencies=["src.features", "src.models", "src.data", "src.analysis.enso_phase"],
        related_tools=["load_enso_data", "load_user_enso"],
    ),
    # --- visualization ---
    "visualization.plots": ComponentInfo(
        key="visualization.plots",
        path="src/visualization/plots.py",
        layer="viz",
        responsibility=(
            "四张图：Niño3.4 时间序列(带±0.5阈值线)、观测vs预测(指定lead+model)、各模型RMSE柱状图、"
            "ENSO相位散点(按相位着色)。统一 Agg 后端、dpi=150。"
        ),
        key_symbols=["plot_enso_timeseries", "plot_observed_vs_predicted", "plot_enso_rmse_by_model", "plot_enso_phase_timeline"],
        dependencies=["matplotlib", "pandas", "src.analysis.enso_phase"],
        related_tools=["plot_enso_timeseries", "plot_observed_vs_predicted", "plot_rmse_by_model", "plot_phase_timeline"],
    ),
    # --- web ---
    "web.app": ComponentInfo(
        key="web.app",
        path="src/web/app.py",
        layer="web",
        responsibility=(
            "Streamlit 入口：侧栏选 LLM 后端(DeepSeek/GLM)+key+上传CSV；session_state 持 messages、ctx、tools；"
            "每轮 run_turn 驱动；工具调用渲染为折叠块、图片内联进对话；history 过长触发 summarize_old_messages。无 key 则禁用输入。"
        ),
        key_symbols=["main", "_resolve_client", "_handle_uploaded_csv", "_new_figures", "_render_tool_step"],
        dependencies=["streamlit", "src.agent.run_turn", "src.agent.tools", "src.agent.client", "src.web.chat_helpers"],
        related_tools=[],
    ),
    "web.chat_helpers": ComponentInfo(
        key="web.chat_helpers",
        path="src/web/chat_helpers.py",
        layer="web",
        responsibility=(
            "UI 纯函数(无 streamlit 依赖，可单测)：SYSTEM_PROMPT(工具选择规则/可信度/回复格式/报告/润色约束)、"
            "init_messages/append_user/parse_tool_step/should_summarize/hint_no_key。"
        ),
        key_symbols=["SYSTEM_PROMPT", "init_messages", "append_user", "parse_tool_step", "should_summarize", "hint_no_key"],
        dependencies=["src.agent.summarizer"],
        related_tools=[],
    ),
    # --- reports ---
    "reports.forecast_report": ComponentInfo(
        key="reports.forecast_report",
        path="src/reports/forecast_report.py",
        layer="reports",
        responsibility=(
            "论文体报告确定性拼装：摘要/引言/方法/结果/结论，数值全从 ctx 真实结果抽取、LLM 不填数。"
            "含 diff_numbers 数值守恒校验器，供 accept_report_polish 拦截 LLM 改数。"
            "训练细节、外源指数含义、图表说明、数据/代码可用性声明皆写进报告。"
        ),
        key_symbols=["generate_forecast_report", "diff_numbers", "extract_numbers", "ReportBundle", "FORECAST_REPORTS_DIR"],
        dependencies=["src.config", "src.agent.tools", "pandas"],
        related_tools=["write_forecast_report", "read_report", "accept_report_polish"],
    ),
    # --- scripts ---
    "scripts.train_cnn_lstm": ComponentInfo(
        key="scripts.train_cnn_lstm",
        path="scripts/train_cnn_lstm.py",
        layer="scripts",
        responsibility="一次性离线训练 CNN-LSTM：读 SODA train/label.nc → train_cnn_lstm → 写 weights/cnn_lstm_soda.pth + 每lead metrics JSON。",
        key_symbols=["main"],
        dependencies=["torch", "src.models.cnn_lstm"],
        related_tools=[],
    ),
    "scripts.run_hindcast": ComponentInfo(
        key="scripts.run_hindcast",
        path="scripts/run_hindcast.py",
        layer="scripts",
        responsibility="一次性 SODA 域回算：用训练好的 CNN-LSTM 在测试窗跑 ACC 对 Persistence，写 cnn_lstm_hindcast.json(并出图)。",
        key_symbols=["main"],
        dependencies=["src.models.hindcast"],
        related_tools=["report_hindcast_skill"],
    ),
    "scripts.run_realtime_hindcast": ComponentInfo(
        key="scripts.run_realtime_hindcast",
        path="scripts/run_realtime_hindcast.py",
        layer="scripts",
        responsibility="一次性 realtime 跨域回算：在 OISST/GODAS/NCEP 域评估，写 cnn_lstm_realtime_hindcast.json——唯一能评判 realtime 预测的 ACC 来源。",
        key_symbols=["main"],
        dependencies=["src.models.realtime_hindcast"],
        related_tools=["report_realtime_skill"],
    ),
    "scripts.build_climatology": ComponentInfo(
        key="scripts.build_climatology",
        path="scripts/build_climatology.py",
        layer="scripts",
        responsibility="一次性预置气候态：按 1991–2020 逐月均值算 OISST/GODAS/NCEP 的 .npz，供 realtime_fetch 反距平化。",
        key_symbols=["main"],
        dependencies=["xarray", "src.data.climatology"],
        related_tools=["forecast_cnn_lstm(mode='realtime')"],
    ),
}


def list_components() -> list[str]:
    """All registered component keys, grouped by layer (for the agent's awareness)."""
    by_layer: dict[str, list[str]] = {}
    for info in COMPONENTS.values():
        by_layer.setdefault(info.layer, []).append(info.key)
    out: list[str] = []
    for layer in by_layer:
        out.append(f"[{layer}] " + ", ".join(by_layer[layer]))
    return out


def explain_component(name: str) -> str:
    """Return a structured textual summary of one registered component.

    If ``name`` is empty, lists every component grouped by layer so the agent
    can pick. Unknown names return a helpful error listing candidates — never
    an empty string.
    """
    if not name:
        return "已注册组件（按层分组）：\n" + "\n".join(list_components())
    if name in COMPONENTS:
        c = COMPONENTS[name]
        lines = [
            f"## 组件 {c.key}",
            f"- 源文件：{c.path}（层：{c.layer}）",
            f"- 职责：{c.responsibility}",
            f"- 关键符号：{', '.join(c.key_symbols)}",
            f"- 依赖：{', '.join(c.dependencies)}",
            f"- 关联工具：{', '.join(c.related_tools) if c.related_tools else '—'}",
            "",
            "如需讲到具体函数/类的源代码，调用 read_source(file_path=\""
            + _PROJECT_ROOT.as_posix() + "/" + c.path
            + "\", symbol=\"<符号名>\") 取真实代码（带行号、可截取）。",
        ]
        return "\n".join(lines)
    # Unknown — suggest close matches by prefix/substring.
    cands = [k for k in COMPONENTS if name in k or k in name]
    hint = ("\n相近匹配：" + ", ".join(cands)) if cands else ""
    return (
        f"未知组件 '{name}'。先调用 explain_component(name=\"\") 列出全部注册组件。{hint}"
    )


# --- read_source ---


def _sandbox(real: Path) -> Path:
    """Resolve a path against the project root and forbid traversal escapes."""
    root = _PROJECT_ROOT.resolve()
    p = (root / real).resolve() if not real.is_absolute() else real.resolve()
    try:
        p.relative_to(root)
    except ValueError as exc:  # paths outside the repo are refused
        raise ValueError(f"path outside project root: {real}") from exc
    return p


def _locate_symbol(lines: list[str], symbol: str) -> tuple[int, int] | None:
    """Find the def/class block for ``symbol`` (incl. its decorators) + its end.

    Returns 1-indexed (start, end_inclusive), or None if not found.

    Two corrections vs. a naive "first def/class line → next outdented line":

    * **Start includes decorators.** A line like ``@dataclass`` directly above
      ``class Foo:`` is part of the definition, so the returned block starts at
      the decorator (walking up over consecutive ``@...`` lines). Otherwise the
      agent would read "class Tool:" without seeing it is a dataclass.
    * **End is the next top-level ``def``/``class`` (or a decorator line that
      introduces one), not a bare module-level assignment.** A regex that treats
      ``SOME_CONST = …`` as a definition boundary wrongly inflates the block
      with the gap before the next real def. Decorators above a *later* def are
      treated as the boundary so they belong to that def, not the current one.
    """
    pat = re.compile(r"^(def|class)\s+" + re.escape(symbol) + r"\b")
    head = None
    for i, ln in enumerate(lines):
        if pat.match(ln):
            head = i
            break
    if head is None:
        return None
    # Walk up over consecutive decorator lines (@ at col 0) just above the head.
    start = head
    while start - 1 >= 0 and lines[start - 1].startswith("@"):
        start -= 1

    # End = first subsequent line at col 0 that is a def/class OR a decorator
    # introducing the next def/class. EOF otherwise. Bare top-level assignments
    # (SOME_CONST = ...) are NOT a boundary — they live between defs.
    end = len(lines) - 1
    for j in range(head + 1, len(lines)):
        ln = lines[j]
        if not ln or ln[0].isspace():
            continue
        if re.match(r"^(def|class)\s+\w", ln) or ln.startswith("@"):
            end = j - 1
            break
    # Trim trailing blank/comment lines so a separator block ("# ---") between
    # this def and the next is not wrongly bundled into the symbol's block.
    while end > head and (
        not lines[end].strip()
        or lines[end].lstrip().startswith("#")
    ):
        end -= 1
    return start + 1, end + 1  # to 1-indexed inclusive


def read_source(
    file_path: str,
    symbol: str | None = None,
    start: int | None = None,
    end: int | None = None,
) -> str:
    """Read a real file from the repo with line numbers; optionally locate a symbol.

    Args:
        file_path: repo-relative or absolute path; sandboxed to the project root.
        symbol: if given, locate ``def symbol``/``class symbol`` and return just
            that block (1-indexed start line reported). Overrides start/end.
        start, end: 1-indexed inclusive line range to return. If only ``start``
            is given, returns that single line. If none given, returns the whole
            file (subject to MAX_SOURCE_CHARS).

    Returns:
        Numbered source text (``NNN:\\t<line>``). Hard-capped at
        MAX_SOURCE_CHARS with a tail marker so a 1500-line module never floods
        the context.
    """
    try:
        raw_path = _sandbox(Path(file_path))
    except ValueError as exc:
        return f"Error: {exc}"
    if not raw_path.is_file():
        return f"Error: file not found: {file_path}"
    try:
        text = raw_path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 — binary/encoding errors must surface
        return f"Error reading file: {exc.__class__.__name__}: {exc}"

    lines = text.splitlines()
    rel = raw_path.relative_to(_PROJECT_ROOT).as_posix() if str(raw_path).startswith(str(_PROJECT_ROOT)) else str(raw_path)

    if symbol:
        loc = _locate_symbol(lines, symbol)
        if loc is None:
            return f"Error: symbol '{symbol}' not found in {file_path}."
        s, e = loc
        chosen = lines[s - 1 : e]
        header = f"{rel} :: symbol '{symbol}' (lines {s}-{e}, {e - s + 1} 行):"
    elif start is not None or end is not None:
        s = max(1, int(start or 1))
        e = min(len(lines), int(end or s))
        chosen = lines[s - 1 : e]
        header = f"{rel} (lines {s}-{e}):"
    else:
        chosen = lines
        header = f"{rel} ({len(lines)} 行):"

    numbered = "\n".join(f"{i+1}:\t{ln}" for i, ln in enumerate(chosen))
    out = f"{header}\n{numbered}"
    if len(out) > MAX_SOURCE_CHARS:
        out = out[:MAX_SOURCE_CHARS] + f"\n…(截断: 已达 {MAX_SOURCE_CHARS} 字符上限;如需更多改用 start/end 分段)"
    return out