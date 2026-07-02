import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
OUTPUTS_DIR = REPORTS_DIR / "outputs"

NOAA_ENSO_RAW_PATH = RAW_DATA_DIR / "noaa_nino34_raw.txt"
NOAA_ENSO_PROCESSED_PATH = PROCESSED_DATA_DIR / "noaa_nino34.csv"
# Default NOAA/PSL Niño3.4 source. Override via the NOAA_NINO34_URL env var to
# point at a mirror (e.g. when psl.noaa.gov is unreachable). For HTTP(S) proxying
# without changing the URL, set HTTPS_PROXY instead — urllib honors it.
NOAA_NINO34_URL_ENV = "NOAA_NINO34_URL"
DEFAULT_NOAA_NINO34_URL = os.environ.get(
    NOAA_NINO34_URL_ENV,
    "https://psl.noaa.gov/data/timeseries/month/data/nino34.long.anom.data",
)

# --- Exogenous climate indices (PSL monthly timeseries, same ASCII format) ---
# SOI (Southern Oscillation Index, atmospheric ENSO precursor) and Niño1+2
# (eastern-Pacific SST, upwelling region precursor). WWV has no reliable static
# source (CPC unreachable, PMEL file removed), so Niño1+2 stands in for the
# ocean-subsurface precursor role. Override each via its own env var if mirrored.
SOI_URL_ENV = "SOI_URL"
DEFAULT_SOI_URL = os.environ.get(
    SOI_URL_ENV, "https://psl.noaa.gov/data/timeseries/month/data/soi.long.data"
)
NINO12_URL_ENV = "NINO12_URL"
DEFAULT_NINO12_URL = os.environ.get(
    NINO12_URL_ENV, "https://psl.noaa.gov/data/timeseries/month/data/nino12.long.data"
)

# --- Data-driven lead confidence (enhanced track) ---
# Replace the hard-coded 7/12-month thresholds with per-lead ACC buckets: an
# Anomaly Correlation Coefficient below ACC_LOW_CONF flags a lead as indicative
# only; below ACC_REFUSE the model refuses to predict. Empirical values, tunable.
ACC_LOW_CONF = 0.5
ACC_REFUSE = 0.3

DEFAULT_LEADS = (1, 3, 6)
DEFAULT_RANDOM_SEED = 42

# --- DeepSeek LLM client (agent layer) ---
# DeepSeek exposes an OpenAI-compatible chat-completions endpoint.
# deepseek-chat supports function calling; deepseek-reasoner does NOT.
DEEPSEEK_API_URL = "https://api.deepseek.com"
DEEPSEEK_CHAT_PATH = "/chat/completions"
DEEPSEEK_MODEL = "deepseek-v4-pro"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
DEEPSEEK_BASE_URL_ENV = "DEEPSEEK_BASE_URL"
DEEPSEEK_MODEL_ENV = "DEEPSEEK_MODEL"
AGENT_REQUEST_TIMEOUT = 120.0

# --- GLM (Zhipu BigModel) LLM client ---
# OpenAI-compatible endpoint; supports function calling with the same tools/
# tool_calls/tool_call_id schema as DeepSeek, so the agent loop is unchanged.
GLM_API_URL = "https://open.bigmodel.cn/api/paas/v4"
GLM_CHAT_PATH = "/chat/completions"
GLM_MODEL = "glm-4.6"
GLM_API_KEY_ENV = "GLM_API_KEY"

# --- Agent loop robustness ---
# _chat_with_retry retries transient DeepSeek errors (429/5xx/network) with
# exponential backoff; AGENT_MAX_RETRIES does not include the initial attempt.
AGENT_MAX_RETRIES = 3
AGENT_RETRY_BASE_DELAY = 0.5
AGENT_RETRY_MAX_DELAY = 8.0
# Consecutive identical (tool, arguments) calls that trip early loop termination.
AGENT_LOOP_LIMIT = 3
# run_turn: max tool calls within a single turn before giving control back.
AGENT_MAX_STURNS = 15

# --- Data freshness (预测时自查) ---
# ENSO is monthly; NOAA/PSL publishes the prior month's value mid-late the
# following month. So "data_through" lagging more than this many months behind
# "now" means the series is stale and a refresh is warranted before trusting a
# forecast (the cutoff month is the forecast baseline). 2 = "if the latest
# value is older than ~2 months, the data is stale". Tunable.
ENSO_STALE_MONTHS = 2
