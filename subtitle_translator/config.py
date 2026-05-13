from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


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


def _env_path(name: str) -> Optional[Path]:
    raw = os.getenv(name, "").strip()
    return Path(raw).expanduser() if raw else None


def _extract_term_values(value: Any) -> list[str]:
    terms: list[str] = []
    if isinstance(value, str):
        terms.extend(term.strip() for term in value.split(",") if term.strip())
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                terms.append(item.strip())
            elif isinstance(item, dict):
                term = str(item.get("term") or item.get("source") or "").strip()
                if term:
                    terms.append(term)
    elif isinstance(value, dict):
        for key in ("approved_terms", "allowed_terms", "terms"):
            if key in value:
                terms.extend(_extract_term_values(value[key]))
        for key, item in value.items():
            if key in {"approved_terms", "allowed_terms", "terms"}:
                continue
            if isinstance(item, (list, dict)):
                terms.extend(_extract_term_values(item))
    return terms


def _load_term_list(path: Optional[Path]) -> list[str]:
    if not path or not path.exists():
        return []
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    try:
        return _extract_term_values(json.loads(raw))
    except json.JSONDecodeError:
        terms = []
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            terms.append(line.split("\t", 1)[0].split(",", 1)[0].strip())
        return [term for term in terms if term]


def _load_english_fallback_terms(path: Optional[Path]) -> list[dict[str, Any]]:
    if not path or not path.exists():
        return []
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    payload = json.loads(raw)
    rows = payload.get("fallback_terms", payload.get("terms", payload)) if isinstance(payload, dict) else payload
    if isinstance(rows, dict):
        rows = [{"source": source, "aliases": aliases} for source, aliases in rows.items()]
    if not isinstance(rows, list):
        return []
    rules: list[dict[str, Any]] = []
    for item in rows:
        if isinstance(item, str):
            source = item.strip()
            aliases: list[str] = []
        elif isinstance(item, dict):
            source = str(item.get("source") or item.get("term") or "").strip()
            raw_aliases = item.get("aliases") or item.get("korean_aliases") or item.get("targets") or []
            if isinstance(raw_aliases, str):
                aliases = [alias.strip() for alias in raw_aliases.split(",") if alias.strip()]
            elif isinstance(raw_aliases, list):
                aliases = [str(alias).strip() for alias in raw_aliases if str(alias).strip()]
            else:
                aliases = []
        else:
            continue
        if source:
            rules.append({"source": source, "aliases": aliases})
    return rules


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
    repair_policy: str
    phase1_prompt_profile: str
    translation_context: str
    translation_style: str
    use_context_window: bool
    repair_enabled: bool
    glossary_log_path: Optional[Path]
    metrics_log_path: Optional[Path]
    glossary_max_terms: int
    request_timeout: int
    request_max_attempts: int
    request_backoff_min_seconds: float
    request_backoff_max_seconds: float
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
    wrap_policy: str
    allowed_english_terms: list[str]
    approved_english_terms_path: Optional[Path]
    english_fallback_map_path: Optional[Path]
    english_fallback_terms: list[dict[str, Any]]
    english_residual_policy: str


def load_runtime_config(glossary_log_path: Optional[str] = None) -> RuntimeConfig:
    env_glossary_path = os.getenv("SRT_GLOSSARY_LOG_PATH", "translation_artifacts/glossary.jsonl").strip()
    resolved_glossary = glossary_log_path if glossary_log_path is not None else env_glossary_path
    glossary_path = Path(resolved_glossary).expanduser() if resolved_glossary else None
    env_metrics_path = os.getenv("SRT_METRICS_LOG_PATH", "translation_artifacts/run_metrics.jsonl").strip()
    metrics_path = Path(env_metrics_path).expanduser() if env_metrics_path else None
    approved_english_terms_path = _env_path("SRT_APPROVED_ENGLISH_TERMS_PATH")
    english_fallback_map_path = _env_path("SRT_ENGLISH_FALLBACK_MAP_PATH")
    env_allowed_english_terms = [
        term.strip()
        for term in os.getenv("SRT_ALLOWED_ENGLISH_TERMS", "PyTorch,NumPy,ResNet,ImageNet,Stanford,CS231n").split(",")
        if term.strip()
    ]
    approved_english_terms = _load_term_list(approved_english_terms_path)
    english_fallback_terms = _load_english_fallback_terms(english_fallback_map_path)
    allowed_english_terms = [
        term.casefold()
        for term in dict.fromkeys(
            env_allowed_english_terms
            + approved_english_terms
            + [str(rule.get("source", "")) for rule in english_fallback_terms]
        )
        if term
    ]
    block_min_cues = max(1, _env_int("SRT_BLOCK_MIN_CUES", 2))
    block_max_cues = max(block_min_cues, _env_int("SRT_BLOCK_MAX_CUES", 4))
    return RuntimeConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        phase1_model=_resolve_phase1_model(),
        repair_model=(os.getenv("SRT_REPAIR_MODEL", "gpt-4o").strip() or "gpt-4o"),
        phase1_temperature=max(0.0, _env_float("SRT_PHASE1_TEMPERATURE", 0.1)),
        repair_temperature=max(0.0, _env_float("SRT_REPAIR_TEMPERATURE", 0.1)),
        repair_policy=(os.getenv("SRT_REPAIR_POLICY", "baseline").strip() or "baseline"),
        phase1_prompt_profile=(os.getenv("SRT_PHASE1_PROMPT_PROFILE", "fragment_preserving_v2").strip() or "fragment_preserving_v2"),
        translation_context=os.getenv("SRT_TRANSLATION_CONTEXT", "").strip(),
        translation_style=os.getenv("SRT_TRANSLATION_STYLE", "").strip(),
        use_context_window=_resolve_context_window_flag(),
        repair_enabled=_env_flag("SRT_ENABLE_REPAIR", True),
        glossary_log_path=glossary_path,
        metrics_log_path=metrics_path,
        glossary_max_terms=max(1, _env_int("SRT_GLOSSARY_MAX_TERMS", 12)),
        request_timeout=max(1, _env_int("SRT_REQUEST_TIMEOUT", 120)),
        request_max_attempts=max(1, _env_int("SRT_REQUEST_MAX_ATTEMPTS", 6)),
        request_backoff_min_seconds=max(0.1, _env_float("SRT_REQUEST_BACKOFF_MIN_SECONDS", 1.0)),
        request_backoff_max_seconds=max(0.1, _env_float("SRT_REQUEST_BACKOFF_MAX_SECONDS", 30.0)),
        phase1_max_retries=max(1, _env_int("SRT_PHASE1_MAX_RETRIES", 2)),
        phase2_max_repairs=max(1, _env_int("SRT_PHASE2_MAX_REPAIRS", 1)),
        max_split_depth=max(0, _env_int("SRT_MAX_SPLIT_DEPTH", 4)),
        block_min_cues=block_min_cues,
        block_max_cues=block_max_cues,
        block_max_duration_ms=max(1000, _env_int("SRT_BLOCK_MAX_DURATION_MS", 6500)),
        block_max_source_chars=max(40, _env_int("SRT_BLOCK_MAX_SOURCE_CHARS", 160)),
        block_max_gap_ms=max(0, _env_int("SRT_BLOCK_MAX_GAP_MS", 800)),
        max_chars_per_line=max(8, _env_int("SRT_MAX_CHARS_PER_LINE", 28)),
        max_lines_per_cue=max(1, _env_int("SRT_MAX_LINES_PER_CUE", 2)),
        max_cps=max(1.0, _env_float("SRT_MAX_CPS", 18.0)),
        wrap_policy=(os.getenv("SRT_WRAP_POLICY", "baseline").strip() or "baseline"),
        allowed_english_terms=allowed_english_terms,
        approved_english_terms_path=approved_english_terms_path,
        english_fallback_map_path=english_fallback_map_path,
        english_fallback_terms=english_fallback_terms,
        english_residual_policy=(
            os.getenv("SRT_ENGLISH_RESIDUAL_POLICY", "coarse").strip() or "coarse"
        ),
    )


def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError("must be an integer >= 1")
    return ivalue
