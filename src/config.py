import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

NOAA_ENSO_RAW_PATH = RAW_DATA_DIR / "noaa_nino34_raw.txt"
NOAA_ENSO_PROCESSED_PATH = PROCESSED_DATA_DIR / "noaa_nino34.csv"
# Default NOAA/PSL Niño3.4 source. Override via the NOAA_NINO34_URL env var to
# point at a mirror (e.g. when psl.noaa.gov is unreachable). For HTTP(S) proxying
# without changing the URL, set HTTPS_PROXY instead — urllib honors it.
NOAA_NINO34_URL_ENV = "NOAA_NINO34_URL"
DEFAULT_NOAA_NINO34_URL = os.environ.get(
    NOAA_NINO34_URL_ENV,
    "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/nino34.long.anom.data",
)

DEFAULT_LEADS = (1, 3, 6)
DEFAULT_RANDOM_SEED = 42

# --- DeepSeek LLM client (agent layer) ---
# DeepSeek exposes an OpenAI-compatible chat-completions endpoint.
# deepseek-chat supports function calling; deepseek-reasoner does NOT.
DEEPSEEK_API_URL = "https://api.deepseek.com"
DEEPSEEK_CHAT_PATH = "/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
DEEPSEEK_BASE_URL_ENV = "DEEPSEEK_BASE_URL"
DEEPSEEK_MODEL_ENV = "DEEPSEEK_MODEL"
AGENT_REQUEST_TIMEOUT = 120.0

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
