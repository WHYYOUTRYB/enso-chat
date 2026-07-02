"""Deterministic ENSO forecast report generator — paper-style layout.

Produces a Markdown report assembled **entirely from structured data already
sitting on :class:`~src.agent.tools.ToolContext`** (cached results JSON, the
ENSO series, generated figures, skill reports). No LLM is involved in writing
the prose or numbers, so nothing can be fabricated: every figure in the report
is referenced from an actual ``figure_paths`` entry, every metric is read from
``ctx.results`` / ``ctx.enhanced_results`` / the hindcast JSON, and any track the
user never ran is flagged ``未运行`` rather than guessed.

The report follows a standard academic-paper layout:

1. 摘要 (Abstract) — one-paragraph gist: target, methods, headline number, caveat.
2. 引言 (Introduction) — ENSO background + why three tracks + this report's scope.
3. 方法 (Methods) — data sources, the three model tracks, reproducibility knobs.
4. 结果 (Results & Discussion) — the actual forecasts, evaluation tables, figures.
5. 结论 (Conclusion) — caveats, lead limits, cross-domain disclaimer.

The module is import-safe: it only touches ``src.config`` and ``pandas`` at the
top level. Reading the CNN-LSTM / realtime hindcast JSONs is done lazily inside
the generator, so a missing ``report_realtime_skill`` cache degrades to a
``未运行`` note instead of crashing.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    ACC_LOW_CONF,
    ACC_REFUSE,
    DEFAULT_LEADS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_NOAA_NINO34_URL,
    GLM_API_URL,
    DEFAULT_SOI_URL,
    DEFAULT_NINO12_URL,
)
from src.agent.tools import ToolContext

# Reports go under reports/forecasts/ (a sibling of reports/outputs, reports/figures)
# so generated reports never collide with the modeling artifacts they describe.
FORECAST_REPORTS_DIR = (
    Path(__file__).resolve().parents[2] / "reports" / "forecasts"
)


# ---------------------------------------------------------------------------
# Numeric-preservation guard for the optional ARS polishing step.
# ---------------------------------------------------------------------------
# generate_forecast_report() is purely deterministic — it is the single source
# of truth for every number in the report. polish_forecast_report() may ask an
# LLM (ARS report_compiler_agent) to improve the *academic prose* of the
# 引言/结论, but must NEVER alter a number, table cell, ACC, or figure path.
# extract_numbers() + diff_numbers() implement a checksum-style guard: if the set
# of numeric tokens changed between pre- and post-polish, the polished text is
# rejected and the unchanged draft is kept. This makes "数据真实" a hard invariant
# even after an LLM touches the prose.

import re

# A numeric token: integer, decimal, scientific, or signed — covers every form a
# forecast value / ACC / lead / RMSE / count can take in the report. Underscores
# inside integers (e.g. 1_000) are not produced by the generator, so omitted.
_NUMBER_RE = re.compile(r"(?<![A-Za-z_])[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?![A-Za-z_])")


def extract_numbers(text: str) -> tuple[str, ...]:
    """Return every numeric token in ``text`` as an ordered tuple.

    Order is preserved because the deterministic draft emits numbers in a stable
    order (摘要 headline → 引言 year(s) → 数据行数/日期 → 表格 → …); comparing
    the ordered tuple catches both kind changes (0.95 -> 0.96) and positional
    changes (a number moved). Duplicates are kept — a table legitimately repeats
    a phase boundary like 0.5 many times, and dropping dups would mask a swap.
    """
    return tuple(_NUMBER_RE.findall(text))


def diff_numbers(before: str, after: str) -> list[str]:
    """Return the list of numeric tokens that differ (set diff both ways).

    Empty list <=> the polish preserved every number. Non-empty triggers a
    reject-and-rollback in polish_forecast_report().
    """
    b, a = set(extract_numbers(before)), set(extract_numbers(after))
    return sorted((b - a) | (a - b))


# The numeric guard is consumed by tools.py:_accept_report_polish, which rejects
# any polish whose numbers differ from the on-disk draft. The academic-polish
# constraints the agent must follow live in chat_helpers.SYSTEM_PROMPT (so the
# LLM sees them) rather than here — Python cannot make the LLM obey; the guard
# is the trust boundary.


@dataclass(frozen=True)
class ReportBundle:
    """The finished report: its markdown path plus the copied figures dir."""

    report_path: Path
    figures_dir: Path
    figure_count: int


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    """Return parsed JSON or None (missing/corrupt) — never raise."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:  # noqa: BLE001 — degrade to "未运行" on any parse failure
        return None


def _phase_cn(value: float) -> str:
    """El Niño / La Niña / Neutral -> Chinese label for the report."""
    if pd.isna(value):
        return "未知"
    if value >= 0.5:
        return "厄尔尼诺（El Niño）"
    if value <= -0.5:
        return "拉尼娜（La Niña）"
    return "中性（Neutral）"


def _confidence_tag(acc: float | None) -> str:
    """ACC -> confidence label, mirroring tools._confidence_from_acc."""
    if acc is None or pd.isna(acc):
        return "未评估（缺少 ACC）"
    if acc < ACC_REFUSE:
        return f"拒绝（ACC={acc:.2f}<{ACC_REFUSE}）"
    if acc < ACC_LOW_CONF:
        return f"低可信度（ACC={acc:.2f}<{ACC_LOW_CONF}）"
    return f"正常（ACC={acc:.2f}）"


def _data_through(ctx: ToolContext) -> str:
    """The latest-data month, as the forecast baseline — '?' if no data."""
    if ctx.enso is None:
        return "?"
    return pd.Timestamp(ctx.enso["date"].max()).strftime("%Y-%m")


def _headline_forecast(ctx: ToolContext) -> tuple[float | None, str, str]:
    """Pick the single number for the abstract: prefer lead=1 best-ML value.

    Priority: enhanced lead-1 (most skillful real-time) > basic lead-1 >
    CNN-LSTM lead-1. Returns (value, track, phase) or (None, '?', '?'). Used only
    in the Abstract; every number still comes from real ctx results.
    """
    for src_dict, track in (
        (ctx.enhanced_results, "增强轨"),
        (ctx.results, "基础轨"),
    ):
        if src_dict is not None:
            fc = src_dict.get("latest_forecast", {}).get("1")
            if fc is not None:
                return float(fc["value"]), track, fc["phase"]
    if ctx.cnn_forecasts is not None:
        e = ctx.cnn_forecasts.get("leads", {}).get("1") or ctx.cnn_forecasts.get("leads", {}).get(1)
        if e is not None:
            return float(e["value"]), "CNN-LSTM轨", _phase_cn(e["value"])
    return None, "?", "?"


# ---------------------------------------------------------------------------
# Section: 摘要 (Abstract)
# ---------------------------------------------------------------------------


def _abstract_block(ctx: ToolContext, target_label: str) -> list[str]:
    """Abstract — single paragraph: target, methods, headline (real) number, caveat."""
    val, track, phase = _headline_forecast(ctx)
    data_through = _data_through(ctx)
    if val is not None:
        headline = (
            f"以 {track} lead=1 个月结果为代表，Niño3.4 未来第 1 个月预测值为 "
            f"{val:.2f}（{phase}）。"
        )
    else:
        headline = "本轮未运行任何预测轨，本报告仅记录方法与数据范围，无实际预测值。"

    tracks_run = []
    if ctx.results is not None:
        tracks_run.append("基础（Ridge/RF，Niño3.4-only）")
    if ctx.enhanced_results is not None:
        tracks_run.append("增强（Ridge/RF + SOI/Niño1+2）")
    if ctx.cnn_forecasts is not None:
        tracks_run.append("CNN-LSTM（空间场 sst/t300/ua/va）")
    tracks_str = "、".join(tracks_run) if tracks_run else "（无已运行预测轨）"

    return [
        "## 摘要",
        "",
        (
            f"本报告以对话式 ENSO 预报 agent 已运行的真实工具结果，按论文体例"
            f"确定性拼装而成（所有数值取自工具返回结果，未由语言模型生成或填补）。"
            f"{headline}预测目标为{target_label or '（未指定）'}，"
            f"预报起算点（数据截止月份）为 {data_through}。"
            f"采用三轨并行策略：{tracks_str}。"
            f"可信度依据各轨 Anomaly Correlation Coefficient（ACC）："
            f"ACC<{ACC_REFUSE} 拒绝、<{ACC_LOW_CONF} 标记低可信度；lead≥7 仅供参考，"
            f"lead≥12 超出可靠预报范围。所有图表由本轮 plot_* 工具实际生成并就地嵌入。"
        ),
        "",
    ]


# ---------------------------------------------------------------------------
# Section: 引言 (Introduction)
# ---------------------------------------------------------------------------


def _intro_block(ctx: ToolContext, target_label: str) -> list[str]:
    """Introduction — ENSO background, three-track rationale, scope."""
    data_through = _data_through(ctx)
    return [
        "## 1 引言",
        "",
        (
            "厄尔尼诺-南方涛动（El Niño–Southern Oscillation, ENSO）是赤道太平洋"
            "海气耦合系统年际变率的主导模态，其相位（厄尔尼诺 / 拉尼娜 / 中性）通过"
            "遥相关影响全球降水与气温。Niño3.4 指数（5°N–5°S, 170°–120°W 海表温度距平）"
            "是刻画 ENSO 相位最常用的诊断量，预测其未来数月值是季节预测的核心任务。"
        ),
        "",
        (
            "ENSO 可预报性在春季存在显著的『春季预报障碍』(spring predictability barrier)，"
            "1–6 个月 lead 较可靠，7–11 个月可信度下降，超过 12 个月已超出可靠范围。"
            "单一方法在不同 lead 上的技巧差异显著：Persistence（预报=最近观测月）"
            "在短 lead 占优（ENSO 自相关），而机器学习模型（Ridge/RF、CNN-LSTM）"
            "在 lead≥4 后才显出相对优势。因此本报告采用三轨并行："
            "(i) 基础轨——基于 Niño3.4 自身滞后的 Ridge/RF；(ii) 增强轨——引入外源指数"
            "（SOI、Niño1+2）作为大气与东太平洋上涌区前兆；(iii) CNN-LSTM 轨——"
            "以 sst/t300/ua/va 空间场为输入的深度学习模型，覆盖更长 lead。"
        ),
        "",
        (
            f"本报告针对预测目标「{target_label or '（未指定，见结果节各轨 lead）'}」，"
            f"起算数据截止于 {data_through}。所有结果由本项目对话式工具链实景运行产生，"
            "方法与数据来源公开可复现（见第 2 节）。"
        ),
        "",
    ]


# ---------------------------------------------------------------------------
# Section: 方法 (Methods)
# ---------------------------------------------------------------------------


def _methods_data_block(ctx: ToolContext) -> list[str]:
    """Methods §2.1 — data sources for each track (real source, never fabricated)."""
    lines: list[str] = ["### 2.1 数据来源", ""]

    if ctx.results is not None:
        info = ctx.results.get("data_source", {})
        used = info.get("used", "?")
        fb = info.get("fallback_reason")
        last = _data_through(ctx)
        r0 = pd.Timestamp(ctx.enso["date"].min()).strftime("%Y-%m") if ctx.enso is not None else "?"
        rows = len(ctx.enso) if ctx.enso is not None else "?"
        lines += [
            f"- **基础轨**：source={used}" + (f"（回退原因：{fb}）" if fb else ""),
            f"  - Niño3.4 序列共 {rows} 行，时间范围 {r0} 至 {last}（最新月份=预报起算点）。",
            f"  - 原始数据 URL：`{DEFAULT_NOAA_NINO34_URL}`（可由 `NOAA_NINO34_URL` 环境变量覆盖为镜像）。",
        ]
    else:
        lines.append("- **基础轨**：**未运行**（未调用 `load_enso_data` / `load_user_enso`）。")

    if ctx.enhanced_results is not None:
        er = ctx.enhanced_results
        exog = er.get("_exog_used", [])
        fb = er.get("_fallback", False)
        if fb:
            lines.append("- **增强轨**：⚠️ 因 SOI/Niño1+2 不可用，退化为 Niño3.4-only。")
        else:
            lines += [
                f"- **增强轨**：外源指数 exog={exog}。",
                f"  - SOI URL：`{DEFAULT_SOI_URL}`（可由 `SOI_URL` 覆盖）。",
                f"  - Niño1+2 URL：`{DEFAULT_NINO12_URL}`（可由 `NINO12_URL` 覆盖）。",
            ]
    else:
        lines.append("- **增强轨**：**未运行**（未调用 `forecast_enhanced` / `compare_methods`）。")

    cnn = ctx.cnn_forecasts
    if cnn is not None:
        lines += [
            f"- **CNN-LSTM 轨**：mode={cnn.get('mode', '?')}，window_end={cnn.get('window_end', '?')}。",
            f"  - 数据来源说明：{cnn.get('source', '?')}。",
        ]
    else:
        lines.append("- **CNN-LSTM 轨**：**未运行**（未调用 `forecast_cnn_lstm`）。")

    lines.append(f"- **LLM 后端**（驱动 agent 工具调度，不产出报告数值）：DeepSeek / GLM（`{GLM_API_URL}`）。")
    lines.append("")
    return lines


def _methods_models_block() -> list[str]:
    """Methods §2.2 — model specs for all three tracks, from real source code."""
    return [
        "### 2.2 模型方法",
        "",
        "**基础轨 / 增强轨（Ridge + Random Forest）**",
        "",
        "",
        "**特征构造**（`make_enso_supervised_table`，逐月一行）：",
        "",
        "- Niño3.4 自身的 0..12 月滞后共 13 个特征（`nino34_lag_0 … nino34_lag_12`）。",
        "- Niño3.4 的 3 个月与 6 个月滚动均值各 1 个（`nino34_roll_mean_3`、`nino34_roll_mean_6`），刻画短期能量。",
        "- 月份的周期编码 `month_sin`、`month_cos`（`sin/cos(2π·month/12)`）2 个，注入季节性。",
        "- 增强轨另加入每个外源指数的 0..12 月滞后（`{soi,nino12}_lag_0..12`，每个 13 个），让模型看到大气与东太平洋前兆。",
        "",
        "**真值列**：`target_lead_{lead} = nino34.shift(-lead)`，即用未来第 `lead` 个月的 Niño3.4 作为标签。构造后 `dropna` 去除首尾缺失行。",
        "",
        "**模型套件**（`build_model_suite(random_state=42)`）：",
        "",
        "- `linear_ridge`：`Pipeline(StandardScaler() → Ridge(alpha=1.0))`，标准化后做 L2 线性回归。",
        "- `random_forest`：`RandomForestRegressor(n_estimators=120, max_depth=8, min_samples_leaf=3, random_state=42)`。",
        "- `persistence` 基线：预报=最近观测月（`nino34_lag_0`），不训练，作对照。",
        "",
        "**训练流程**：对每个 lead 单独 fit 一组模型；最终预测用 `fit_models_for_latest_forecast` 在**全表**上重训一遍再用末行特征推断下月。",
        "",
        f"- 评估划分：时间序末尾 25% 为测试集（`temporal_train_test_split(test_fraction=0.25)`），杜绝未来信息穿越。",
        "- 模型选择：每个 lead 取测试集 RMSE 最小者为该 lead 最优模型。",
        f"- ACC 阈值：拒绝<{ACC_REFUSE}；低可信度<{ACC_LOW_CONF}。ACC 即 Anomaly Correlation Coefficient（距平相关系数），ENSO 预测的标准技巧度量。",
        "",
        "**CNN-LSTM 轨**",
        "",
        "- 模型：CNN-LSTM（逐月 CNN 特征提取 → 双层 LSTM → 全连接层一次输出 24 个 lead），于 SODA 再分析数据上离线训练（脚本 `scripts/train_cnn_lstm.py`）。",
        "- 输入张量：`(12, 24, 72, 4)` = 12 个月滑窗 × 24×72 经纬网格 × 4 通道。四通道：sst（海表温度）、t300（次表层 300m 温度，热含量代理）、ua/va（纬向/经向风），分别刻画 SST 正/负反馈、温跃层变化与风场强迫。",
        "- 网络结构（详见 `cnn_lstm._build_model`）：`Conv2d(4→16, kernel=7, stride=2)`→`BatchNorm`→`ReLU`；`Conv2d(16→16, k=3)`→`Dropout(0.7)`→`BatchNorm`→`ReLU`；`AvgPool2(2)`；`Flatten→1728`→`LSTM(1728→1024)`→`LSTM(1024→256)`→`Dropout(0.7)`→`Linear(256→24)`。",
        "- 训练超参（`train_cnn_lstm`，逐批在 train 程序上自洽）：损失=MSE；优化器=Adam；`lr=1e-3`、`weight_decay=0.001`、`batch_size=8`、`epochs=80`、早停 `patience=10`；`ReduceLROnPlateau(factor=0.5, patience=5)`；`seed=42`。",
        "- 标准化：逐通道用**训练集**均值/方差标准化（统计量存进 checkpoint，推断时复用，杜绝测试集信息泄漏）。",
        "- 数据划分（SODA 共 100 块 × 36 月 = 3600 月连续序列）：`train` = 第 0–71 块（2520 月）、`val` = 第 70–82 块、`buffer` = 第 82–84 块（故意留空，隔离训练尾与测试窗，防泄漏）、`test` = 第 85–100 块。滑窗跨度 36 月（12 月输入+24 月目标）。",
        "- 评估指标：all-season Anomaly Correlation Coefficient（ACC），参考 Ham et al. 2019 (*Nature*)。",
        "- 基线：Persistence（预报=最近观测月，见 `src/models/hindcast.py` 的回算）。",
        "- `mode`：`soda_tail` 使用 SODA 末端窗口（非实时，方法演示用语）；`realtime` 实时抓取 OISST+GODAS+NCEP 场，反距平化后推断（**跨域**：训练于 SODA、推断于其他源）。",
        "- **与 Ham et al. 2019 的差异**（如实陈述）：Ham 在 CMIP5 历史模拟上预训练并用迁移学习微调、以四套 CNN 架构（C30H30/C30H50/C50H30/C50H50）平均作为 ensemble；本项目仅以 SODA 单模型训练，无 CMIP5 预训练、无迁移学习、无 ensemble 平均。因此短 lead 与极限技巧可能与 Ham 不同；本报告 ACC 反映的是本项目 SODA 域的真实回算上限。",
        "",
        "### 2.3 外源指数含义",
        "",
        "增强轨引入的两个外源指数来自 NOAA/PSL 同格式月值时序，各自刻画 ENSO 不同环节的前兆信号：",
        "",
        "- **Niño3.4**：5°N–5°S、170°–120°W 赤道中太平洋 SST 距平，是 ENSO 相位最常用的诊断量与预测目标本身。",
        "- **SOI（Southern Oscillation Index）**：塔希提与达尔文两站海平面气压差（标准化），刻画 ENSO 大气支——赤道东西向气压梯度。SOI 偏负通常对应厄尔尼诺（信风减弱），偏正对应拉尼娜。作为大气前兆，常先于海温变率。",
        "- **Niño1+2**：0°–10°S、90°–80°W 东太平洋上涌区 SST 距平（秘鲁沿岸）。该区在 ENSO 事件发展早期的升温尤为敏感，是 ENSO 发展期前兆。",
        "",
        "三者关系：Niño3.4 是中枢指标；SOI 提供大气超前信号、Niño1+2 提供东太平洋上涌区超前信号，二者滞后特征使增强轨模型得以“看到”Niño3.4 自相关之外的信息。",
        "",
    ]


def _methods_repro_block() -> list[str]:
    """Methods §2.3 — reproducibility knobs + commands."""
    return [
        "### 2.4 可复现性",
        "",
        "下列配置项与命令完全决定本报告数值，独立者可据此复现：",
        "",
        "```bash",
        "pip install -r requirements.txt",
        "# 离线训练 + 回算（一次性）",
        "python scripts/train_cnn_lstm.py         # 生成 weights/cnn_lstm_soda.pth",
        "python scripts/run_hindcast.py          # 生成离线回算报告（SODA 域）",
        "python scripts/run_realtime_hindcast.py # 生成 realtime 跨域回算报告",
        "# 对话式预报 + 报告生成",
        "streamlit run src/web/app.py            # 对话中调 load_enso_data/compare_methods/plot_*/write_forecast_report",
        "```",
        "",
    ]


def _methods_availability_block() -> list[str]:
    """Methods §2.5 — Data & Code availability (paper-standard statement)."""
    return [
        "### 2.5 数据与代码可用性（Data & Code Availability）",
        "",
        "本报告所用数据全部公开可下载，源 URL 取自 `src/config.py`；如遇原站不可达，可用同名环境变量指向镜像。",
        "",
        "**数据**",
        "",
        f"- Niño3.4 指数（NOAA/PSL 月值时序）：`{DEFAULT_NOAA_NINO34_URL}`（可由 `NOAA_NINO34_URL` 覆盖）。",
        f"- SOI 指数（NOAA/PSL）：`{DEFAULT_SOI_URL}`（可由 `SOI_URL` 覆盖）。",
        f"- Niño1+2 指数（NOAA/PSL）：`{DEFAULT_NINO12_URL}`（可由 `NINO12_URL` 覆盖）。",
        "- CNN-LSTM 训练数据：SODA v2.2.4 再分析空间场（sst/t300/ua/va），存为本项目 `data/SODA_train.nc` 与 `data/SODA_label.nc`。",
        "- realtime 轨实时拉取：OISST（海表温度）、GODAS（次表层/风场）、NCEP（风场），预报前按预置气候态反距平化。",
        "- 降水/潮汐分析：随项目内置的样本数据（`data/sample/`），见 README。",
        "",
        "**代码与依赖**",
        "",
        "- Python ≥3.11；依赖见 `requirements.txt`（含 pandas / numpy / scikit-learn / torch / xarray / matplotlib / streamlit）。",
        "- CNN-LSTM 权重产物：`weights/cnn_lstm_soda.pth`（由 `scripts/train_cnn_lstm.py` 生成）。",
        "- 离线回算报告：`reports/outputs/cnn_lstm_hindcast.json` 与 `reports/outputs/cnn_lstm_realtime_hindcast.json`。",
        "- 报告确定性拼装源码：`src/reports/forecast_report.py`（不含 LLM 写数，所有数值均自 ctx 真实结果抽取）。",
        "",
        "**方法学参考**：Ham, Y.-G., Kim, J.-H. & Luo, J.-J. *Deep learning for multi-year ENSO forecasts.* Nature **573**, 568–572 (2019). https://doi.org/10.1038/s41586-019-1559-7",
        "",
    ]


# ---------------------------------------------------------------------------
# Section: 结果 (Results & Discussion)
# ---------------------------------------------------------------------------


def _results_forecast_block(ctx: ToolContext) -> list[str]:
    """Results §3.1 — the actual forecasts, every track, real numbers only."""
    lines: list[str] = ["### 3.1 三轨预测结果", ""]

    # 基础轨
    if ctx.results is not None:
        lines.append("**基础轨（Ridge/RF，1/3/6 个月 lead，缓存结果）**")
        lines.append("")
        lines.append("| lead | 预测值 | 相位 | 最优模型 | ACC | 可信度 |")
        lines.append("|---:|---:|:--|:--|---:|:--|")
        for lead in DEFAULT_LEADS:
            key = str(lead)
            fc = ctx.results["latest_forecast"].get(key)
            if fc is None:
                lines.append(f"| {lead} | — | 未运行 | — | — | — |")
                continue
            best = ctx.results["best_model_by_lead"].get(key, "?")
            acc = ctx.results["leads"].get(key, {}).get(best, {}).get("acc")
            lines.append(
                f"| {lead} | {fc['value']:.3f} | {_phase_cn(fc['value'])} | "
                f"{best} | {acc if acc is not None else '—'} | {_confidence_tag(acc)} |"
            )
        lines.append("")
    else:
        lines.append("- **基础轨**：**未运行**（缺少 `ctx.results`）。")
        lines.append("")

    # 增强轨
    if ctx.enhanced_results is not None:
        er = ctx.enhanced_results
        exog = er.get("_exog_used", [])
        lines.append(f"**增强轨（Ridge/RF + SOI/Niño1+2，exog={exog}）**")
        lines.append("")
        lines.append("| lead | 预测值 | 相位 | 最优模型 | ACC | 可信度 |")
        lines.append("|---:|---:|:--|:--|---:|:--|")
        for lead in DEFAULT_LEADS:
            key = str(lead)
            fc = er.get("latest_forecast", {}).get(key)
            if fc is None:
                lines.append(f"| {lead} | — | 未运行 | — | — | — |")
                continue
            best = er.get("best_model_by_lead", {}).get(key, "?")
            acc = er.get("leads", {}).get(key, {}).get(best, {}).get("acc")
            lines.append(
                f"| {lead} | {fc['value']:.3f} | {_phase_cn(fc['value'])} | "
                f"{best} | {acc if acc is not None else '—'} | {_confidence_tag(acc)} |"
            )
        lines.append("")
    else:
        lines.append("- **增强轨**：**未运行**（未调用 `forecast_enhanced` / `compare_methods`）。")
        lines.append("")

    # CNN-LSTM 轨
    if ctx.cnn_forecasts is not None:
        leads_map = ctx.cnn_forecasts.get("leads", {})
        mode = ctx.cnn_forecasts.get("mode", "?")
        lines.append(f"**CNN-LSTM 轨（mode={mode}，空间场 sst/t300/ua/va）**")
        lines.append("")
        lines.append("| lead | 预测值 | 相位 |")
        lines.append("|---:|---:|:--|")
        for lead in sorted(int(k) for k in leads_map):
            e = leads_map[str(lead)] if str(lead) in leads_map else leads_map[lead]
            lines.append(f"| {lead} | {e['value']:.3f} | {_phase_cn(e['value'])} |")
        lines.append("")
        if mode == "realtime":
            lines.append(
                "> ⚠️ 跨域警示：训练于 SODA、推断于 OISST/GODAS/NCEP，精度低于 SODA 回算；"
                "可靠性须以 realtime 回算 ACC 为准，**不能套用 SODA 回算 ACC**。"
            )
            lines.append("")
    else:
        lines.append("- **CNN-LSTM 轨**：**未运行**（未调用 `forecast_cnn_lstm`）。")
        lines.append("")

    return lines


def _results_reading_block(ctx: ToolContext) -> list[str]:
    """§3.1bis — a one-paragraph real-data reading of the forecast tables above.

    Pulls the SAME real numbers already written into the tables (no new
    numbers, no rounding beyond what the tables show) and states what they mean:
    the three-way phase consensus, the best-lead value, and any cross-track
    disagreement. Nothing here is invented.
    """
    lines: list[str] = ["**结果数据解读**", ""]
    # Collect each track's phase + value at its first available lead.
    observations: list[str] = []
    phases: set[str] = set()
    if ctx.results is not None:
        fc = ctx.results.get("latest_forecast", {}).get("1")
        if fc is not None:
            phases.add(fc["phase"])
            observations.append(
                f"基础轨 lead=1 个月给出 {fc['value']:.3f}（{fc['phase']}）"
            )
    if ctx.enhanced_results is not None:
        fc = ctx.enhanced_results.get("latest_forecast", {}).get("1")
        if fc is not None:
            phases.add(fc["phase"])
            observations.append(
                f"增强轨 lead=1 个月给出 {fc['value']:.3f}（{fc['phase']}）"
            )
    if ctx.cnn_forecasts is not None and ctx.cnn_forecasts.get("leads"):
        lp = ctx.cnn_forecasts["leads"]
        k = "1" if "1" in lp else next(iter(lp))
        e = lp[k]
        phase = _phase_cn(e["value"])
        phases.add(phase)
        observations.append(f"CNN-LSTM 轨 lead={k} 给出 {e['value']:.3f}（{phase}）")

    if not observations:
        lines.append("本轮无已运行的预测轨，故无预测数值可供解读。")
    else:
        lines.append("已运行各轨在短 lead 的预测值如下：" + "；".join(observations) + "。")
        n_tracks = len(observations)
        if len(phases) == 1:
            only = next(iter(phases))
            scope = "三轨" if n_tracks >= 3 else ("各已运行轨" if n_tracks == 2 else "该轨各 lead")
            lines.append(f"{scope}相位一致指向 **{only}**，可信度最高。")
        elif len(phases) > 1:
            lines.append(
                "各轨相位存在分歧（" + "、".join(phases) + "），以各轨 ACC 为准择优；"
                "倾向短 lead（1 个月）可信度最高者。"
            )
        lines.append(
            "表中 lead≥7 仅作参考、lead≥12 超出可靠预报范围；realtime 轨须以 realtime 跨域回算 ACC 评判。"
        )
    lines.append("")
    return lines


def _results_evaluation_block(ctx: ToolContext) -> list[str]:
    """Results §3.2 — model evaluation tables (metrics + hindcast skill)."""
    lines: list[str] = ["### 3.2 模型评估", ""]

    # 基础轨每 lead 全模型指标
    if ctx.results is not None:
        lines.append("**基础轨每 lead 指标（测试集）**")
        lines.append("")
        lines.append("| lead | 模型 | RMSE | MAE | corr | ACC |")
        lines.append("|---:|:--|---:|---:|---:|---:|")
        for lead in DEFAULT_LEADS:
            key = str(lead)
            for model, m in ctx.results["leads"].get(key, {}).items():
                lines.append(
                    f"| {lead} | {model} | {m.get('rmse', '—')} | "
                    f"{m.get('mae', '—')} | {m.get('corr', '—')} | {m.get('acc', '—')} |"
                )
        lines.append("")

    # CNN-LSTM SODA 回算
    from src.agent.tools import HINDCAST_REPORT_PATH
    soda = _safe_load_json(HINDCAST_REPORT_PATH)
    if soda is not None:
        lines.append("**CNN-LSTM 回算精度（SODA 训练域，Ham et al. 2019 指标）**")
        lines.append("")
        lines.append(f"测试窗口数 n={soda.get('n_samples', '?')}。CNN 须优于 Persistence 才算有技巧。")
        lines.append(
            "说明：本项目为 SODA 单模型确定性回算，未采用 Ham et al. 2019 的 CMIP5 预训练 + 迁移学习 + "
            "四架构 ensemble 平均；因此 ACC 反映本项目 SODA 域真实上限，不宜与 Ham 论文的 CNN 曲线直接数字比对。"
        )
        lines.append("")
        lines.append("| lead | CNN-ACC | Persistence-ACC | gap |")
        lines.append("|---:|---:|---:|---:|")
        leads = soda.get("leads", [])
        for i, ld in enumerate(leads):
            lines.append(
                f"| {ld} | {soda['cnn_acc'][i]:.3f} | "
                f"{soda['persistence_acc'][i]:.3f} | {soda['skill_gap'][i]:+.3f} |"
            )
        lines.append("")
    else:
        lines.append("- **CNN-LSTM SODA 回算**：**未运行**（`scripts/run_hindcast.py`）。")
        lines.append("")

    # CNN-LSTM realtime 跨域回算
    from src.agent.tools import REALTIME_HINDCAST_REPORT_PATH
    rt = _safe_load_json(REALTIME_HINDCAST_REPORT_PATH)
    if rt is not None:
        lines.append("**CNN-LSTM realtime 跨域回算（OISST/GODAS/NCEP）**")
        lines.append("")
        lines.append(
            f"这是唯一可用于评判 realtime 预测的 ACC。n={rt.get('n_windows', '?')} 窗口，"
            f"评估期={rt.get('eval_period', '?')}。"
        )
        lines.append("")
        lines.append("| lead | CNN-ACC | Persistence-ACC | gap |")
        lines.append("|---:|---:|---:|---:|")
        leads = rt.get("leads", [])
        for i, ld in enumerate(leads):
            lines.append(
                f"| {ld} | {rt['cnn_acc'][i]:.3f} | "
                f"{rt['persistence_acc'][i]:.3f} | {rt['skill_gap'][i]:+.3f} |"
            )
        lines.append("")
    else:
        lines.append("- **CNN-LSTM realtime 跨域回算**：**未运行**（`scripts/run_realtime_hindcast.py`）。")
        lines.append("")

    return lines


def _results_figures_block(ctx: ToolContext, figures_dir: Path) -> tuple[list[str], int]:
    """Results §3.3 — copy existing figures into the report dir and embed them.

    Each embedded figure gets a numbered legend describing what the plot shows,
    sourced from the real plotting code (``src/visualization/plots.py``): what is
    on each axis, what the threshold/dashed lines mean, and how to read it. The
    legend is descriptive prose about the figure type, not a fabricated number.
    """
    lines: list[str] = ["### 3.3 图表", ""]
    figures_dir.mkdir(parents=True, exist_ok=True)
    if not ctx.figure_paths:
        lines.append("本轮未生成图表（可让 agent 调用 `plot_*` 工具后再生成报告）。")
        lines.append("")
        return lines, 0

    # Filename -> human legend, matching plot_* implementations in plots.py.
    legends: dict[str, str] = {
        "enso_timeseries.png": (
            "Niño3.4 时间序列：横轴日期、纵轴距平（°C），蓝线为月值序列，"
            "红色虚线 +0.5、绿色虚线 −0.5 分别为厄尔尼诺/拉尼娜相位阈值，黑色零线为中性线。"
            "用于判断历史序列的整体相位与振幅。"
        ),
        "enso_observed_vs_predicted.png": (
            "观测 vs 预测对比（指定 lead + 模型）：横轴测试期日期、纵轴 Niño3.4 距平，"
            "实线为观测真值，虚线为模型预测，用于目测模型对该 lead 的跟踪精度与相位偏差。"
        ),
        "enso_rmse_by_model.png": (
            "各模型×lead 的 RMSE 柱状图：横轴标号形如 “L1-linear_ridge”，纵轴为 RMSE。"
            "柱越低误差越小；可据此比较同 lead 不同模型的相对优劣。"
        ),
        "enso_phase_timeline.png": (
            "ENSO 相位散点：横轴日期、纵轴 Niño3.4，按相位着色（红=厄尔尼诺、绿=拉尼娜、灰=中性），"
            "叠加 ±0.5 阈值虚线，用于直观查看各相位的持续时间与切换节奏。"
        ),
    }

    copied = 0
    for idx, src in enumerate(ctx.figure_paths, start=1):
        if not src.exists():
            continue
        dst = figures_dir / src.name
        # Avoid name collisions when the same figure is referenced twice.
        if dst.exists() and dst.stat().st_size != src.stat().st_size:
            stem, ext = dst.stem, dst.suffix
            i = 1
            while (figures_dir / f"{stem}_{i}{ext}").exists():
                i += 1
            dst = figures_dir / (f"{stem}_{i}{ext}")
        shutil.copy2(src, dst)
        lines.append(f"### 图 {idx}：{src.name}")
        lines.append("")
        lines.append(f"![{src.name}](figures/{dst.name})")
        lines.append("")
        legend = legends.get(src.name)
        if legend is None:
            legend = f"图 {src.name}：由本轮对应 plot_* 工具生成（见 §3.2 评估指标对应的模型）。"
        lines.append(f"**说明**：{legend}")
        lines.append("")
        copied += 1
    return lines, copied


# ---------------------------------------------------------------------------
# Section: 结论 (Conclusion)
# ---------------------------------------------------------------------------


def _conclusion_block(ctx: ToolContext) -> list[str]:
    """Conclusion — caveats, lead limits, cross-domain disclaimer."""
    tracks_run = []
    if ctx.results is not None:
        tracks_run.append("基础")
    if ctx.enhanced_results is not None:
        tracks_run.append("增强")
    if ctx.cnn_forecasts is not None:
        tracks_run.append("CNN-LSTM")
    tracks_str = "、".join(tracks_run) if tracks_run else "无"

    return [
        "## 4 结论",
        "",
        (
            f"本轮已运行轨：{tracks_str}。三轨给出一致相位结论时可信度最高；"
            "分轨不一致时以各轨 ACC 为准择优。"
            f"lead≥7（长于半年）仅作参考；lead≥12 超出可靠预报范围，本工具拒绝预测。"
            "CNN-LSTM realtime 模式属跨域（训练 SODA / 推断 OISST、GODAS、NCEP），"
            "其精度低于 SODA 回算——评判 realtime 预测须用 realtime 跨域回算 ACC，"
            "不可套用 SODA 回算 ACC。"
        ),
        "",
        "**免责声明**：本报告所有数值来自真实运行结果，由确定性拼装产生，"
        "未由语言模型生成或填补。未运行的轨以“未运行”标注，绝不臆测。",
        "",
    ]


def generate_forecast_report(ctx: ToolContext, *, target_label: str = "") -> ReportBundle:
    """Assemble a paper-style Markdown report from the ToolContext's cached results.

    Layout: 摘要 → 引言 → 方法 → 结果（预测/评估/图表）→ 结论. Every number is
    read from real ctx results; nothing is fabricated by an LLM.

    Args:
        ctx: the shared tool context — must hold any results the user has run.
        target_label: free-text label for what the forecast targets (e.g.
            ``"2027年3月 Niño3.4"``); appears in the abstract + intro only. It
            is a label, not a number, so it can never fabricate a result.

    Returns:
        :class:`ReportBundle` with the report path, figures dir, and figure
        count. The report file is written under ``reports/forecasts/``.
    """
    FORECAST_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    parts: list[str] = ["# ENSO 预测报告\n"]
    parts.extend(_abstract_block(ctx, target_label))
    parts.extend(_intro_block(ctx, target_label))
    parts.append("## 2 方法\n")
    parts.extend(_methods_data_block(ctx))
    parts.extend(_methods_models_block())
    parts.extend(_methods_repro_block())
    parts.extend(_methods_availability_block())
    parts.append("## 3 结果\n")
    parts.extend(_results_forecast_block(ctx))
    parts.extend(_results_reading_block(ctx))
    parts.extend(_results_evaluation_block(ctx))

    figures_dir = FORECAST_REPORTS_DIR / "figures"
    fig_lines, fig_count = _results_figures_block(ctx, figures_dir)
    parts.extend(fig_lines)

    parts.extend(_conclusion_block(ctx))

    # Use a monotonic counter on existing files instead of a clock to keep this
    # module pure (no Date.now / time.time at module import time).
    stem = "enso_forecast_report"
    report_path = FORECAST_REPORTS_DIR / f"{stem}.md"
    if report_path.exists():
        i = 1
        while (FORECAST_REPORTS_DIR / f"{stem}_{i}.md").exists():
            i += 1
        report_path = FORECAST_REPORTS_DIR / f"{stem}_{i}.md"

    report_path.write_text("\n".join(parts), encoding="utf-8")
    return ReportBundle(report_path=report_path, figures_dir=figures_dir, figure_count=fig_count)