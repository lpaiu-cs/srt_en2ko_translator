from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _strip_matching_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            os.environ.setdefault(key, _strip_matching_quotes(value.strip()))


load_dotenv(PROJECT_ROOT / ".env")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


@dataclass
class RuntimeConfig:
    openai_api_key: str
    translation_context: str
    translation_style: str
    use_previous_translation_context: bool
    glossary_log_path: Optional[Path]
    glossary_max_terms: int
    request_timeout: int


def load_runtime_config(glossary_log_path: Optional[str] = None) -> RuntimeConfig:
    env_glossary_path = os.getenv("SRT_GLOSSARY_LOG_PATH", "translation_artifacts/glossary.jsonl").strip()
    resolved_glossary = glossary_log_path if glossary_log_path is not None else env_glossary_path
    glossary_path = Path(resolved_glossary).expanduser() if resolved_glossary else None
    return RuntimeConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        translation_context=os.getenv("SRT_TRANSLATION_CONTEXT", "").strip(),
        translation_style=os.getenv("SRT_TRANSLATION_STYLE", "").strip(),
        use_previous_translation_context=_env_flag("SRT_USE_PREVIOUS_CONTEXT", True),
        glossary_log_path=glossary_path,
        glossary_max_terms=max(1, _env_int("SRT_GLOSSARY_MAX_TERMS", 12)),
        request_timeout=max(1, _env_int("SRT_REQUEST_TIMEOUT", 120)),
    )


def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError("must be an integer >= 1")
    return ivalue
