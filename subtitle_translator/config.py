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


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _resolve_phase1_model() -> str:
    return (
        os.getenv("SRT_PHASE1_MODEL")
        or os.getenv("SRT_OPENAI_MODEL")
        or "gpt-4.1-mini"
    ).strip() or "gpt-4.1-mini"


def _resolve_context_window_flag() -> bool:
    if os.getenv("SRT_USE_CONTEXT_WINDOW") is not None:
        return _env_flag("SRT_USE_CONTEXT_WINDOW", True)
    return _env_flag("SRT_USE_PREVIOUS_CONTEXT", True)


@dataclass
class RuntimeConfig:
    openai_api_key: str
    phase1_model: str
    repair_model: str
    phase1_temperature: float
    repair_temperature: float
    phase1_prompt_profile: str
    translation_context: str
    translation_style: str
    use_context_window: bool
    repair_enabled: bool
    glossary_log_path: Optional[Path]
    metrics_log_path: Optional[Path]
    glossary_max_terms: int
    request_timeout: int
    phase1_max_retries: int
    phase2_max_repairs: int
    max_split_depth: int
    block_min_cues: int
    block_max_cues: int
    block_max_duration_ms: int
    block_max_source_chars: int
    block_max_gap_ms: int
    max_chars_per_line: int
    max_lines_per_cue: int
    max_cps: float
    allowed_english_terms: list[str]


def load_runtime_config(glossary_log_path: Optional[str] = None) -> RuntimeConfig:
    env_glossary_path = os.getenv("SRT_GLOSSARY_LOG_PATH", "translation_artifacts/glossary.jsonl").strip()
    resolved_glossary = glossary_log_path if glossary_log_path is not None else env_glossary_path
    glossary_path = Path(resolved_glossary).expanduser() if resolved_glossary else None
    env_metrics_path = os.getenv("SRT_METRICS_LOG_PATH", "translation_artifacts/run_metrics.jsonl").strip()
    metrics_path = Path(env_metrics_path).expanduser() if env_metrics_path else None
    block_min_cues = max(1, _env_int("SRT_BLOCK_MIN_CUES", 2))
    block_max_cues = max(block_min_cues, _env_int("SRT_BLOCK_MAX_CUES", 4))
    return RuntimeConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        phase1_model=_resolve_phase1_model(),
        repair_model=(os.getenv("SRT_REPAIR_MODEL", "gpt-4o").strip() or "gpt-4o"),
        phase1_temperature=max(0.0, _env_float("SRT_PHASE1_TEMPERATURE", 0.1)),
        repair_temperature=max(0.0, _env_float("SRT_REPAIR_TEMPERATURE", 0.1)),
        phase1_prompt_profile=(os.getenv("SRT_PHASE1_PROMPT_PROFILE", "fragment_preserving_v2").strip() or "fragment_preserving_v2"),
        translation_context=os.getenv("SRT_TRANSLATION_CONTEXT", "").strip(),
        translation_style=os.getenv("SRT_TRANSLATION_STYLE", "").strip(),
        use_context_window=_resolve_context_window_flag(),
        repair_enabled=_env_flag("SRT_ENABLE_REPAIR", True),
        glossary_log_path=glossary_path,
        metrics_log_path=metrics_path,
        glossary_max_terms=max(1, _env_int("SRT_GLOSSARY_MAX_TERMS", 12)),
        request_timeout=max(1, _env_int("SRT_REQUEST_TIMEOUT", 120)),
        phase1_max_retries=max(1, _env_int("SRT_PHASE1_MAX_RETRIES", 2)),
        phase2_max_repairs=max(1, _env_int("SRT_PHASE2_MAX_REPAIRS", 1)),
        max_split_depth=max(0, _env_int("SRT_MAX_SPLIT_DEPTH", 4)),
        block_min_cues=block_min_cues,
        block_max_cues=block_max_cues,
        block_max_duration_ms=max(1000, _env_int("SRT_BLOCK_MAX_DURATION_MS", 6500)),
        block_max_source_chars=max(40, _env_int("SRT_BLOCK_MAX_SOURCE_CHARS", 160)),
        block_max_gap_ms=max(0, _env_int("SRT_BLOCK_MAX_GAP_MS", 800)),
        max_chars_per_line=max(8, _env_int("SRT_MAX_CHARS_PER_LINE", 24)),
        max_lines_per_cue=max(1, _env_int("SRT_MAX_LINES_PER_CUE", 2)),
        max_cps=max(1.0, _env_float("SRT_MAX_CPS", 18.0)),
        allowed_english_terms=[
            term.strip().casefold()
            for term in os.getenv("SRT_ALLOWED_ENGLISH_TERMS", "PyTorch,NumPy,ResNet,ImageNet,Stanford,CS231n")
            .split(",")
            if term.strip()
        ],
    )


def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError("must be an integer >= 1")
    return ivalue
