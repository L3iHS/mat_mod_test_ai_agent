from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path | None = None) -> None:
    """Минимальный загрузчик .env, чтобы не добавлять отдельную зависимость"""
    env_path = path or PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    try:
        from dotenv import load_dotenv as load_python_dotenv

        load_python_dotenv(env_path, override=False)
        return
    except ImportError:
        pass

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Settings:
    llm_api_key: str | None
    llm_base_url: str
    llm_model: str
    github_token: str | None
    data_dir: Path


def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        llm_api_key=os.getenv("LLM_API_KEY") or os.getenv("AGENT_ROUTER_API_KEY"),
        llm_base_url=os.getenv("LLM_BASE_URL", "https://agentrouter.org/v1").rstrip("/"),
        llm_model=os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001"),
        github_token=os.getenv("GITHUB_TOKEN"),
        data_dir=PROJECT_ROOT / "data" / "generated",
    )
