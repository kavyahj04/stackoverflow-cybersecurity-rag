"""All settings in one place.

Secrets and per-environment values come from rag-dev/.env (or real environment
variables in production). Everything else has a default here, so no other
file hard-codes a model name, a database address or a search setting.
"""
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
REPORTS_DIR = ROOT_DIR / "reports"
TEST_SET_PATH = SRC_DIR / "test_set_human.jsonl"


@dataclass(frozen=True)
class Settings:
    database_url: str
    # Must match the model that created the stored question embeddings
    embedding_model: str = "text-embedding-3-large"
    # HNSW search shortlist: 100 gave 99.3% of exact-search top 10 at the same speed
    hnsw_ef_search: int = 100
    # How many threads search returns, and how many answers per thread go into the context
    top_k: int = 10
    max_answers_per_thread: int = 3

    # Generation
    llm_model: str = "gpt-4.1-mini"
    # 0 makes answers as repeatable as possible; set to None for models that reject it (reasoning models)
    llm_temperature: float | None = 0.0
    llm_timeout_s: float = 60.0
    max_answer_tokens: int = 1_000
    # Context cap: trimmed 29 of 485 test queries and never dropped the correct thread
    max_context_tokens: int = 10_000
    # Best-match cosine distance above which we answer "nothing relevant found" without calling the LLM.
    # All 485 real test questions were <= 0.503; 39 of 40 unrelated questions were >= 0.574
    max_distance: float = 0.55


@lru_cache
def get_settings() -> Settings:
    """Read settings once, on first use (not at import time)."""
    load_dotenv(ROOT_DIR / ".env", override=True)
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set. Add it to rag-dev/.env")
    return Settings(
        database_url=database_url,
        embedding_model=os.environ.get("EMBEDDING_MODEL", Settings.embedding_model),
        hnsw_ef_search=int(os.environ.get("HNSW_EF_SEARCH", Settings.hnsw_ef_search)),
        llm_model=os.environ.get("LLM_MODEL", Settings.llm_model),
    )
