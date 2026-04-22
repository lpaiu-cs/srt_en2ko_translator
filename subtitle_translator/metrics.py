from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .models import Cue
from .splitting import ts_to_ms
from .text import normalize_text, warn


@dataclass
class TranslationMetrics:
    blocks_started: int = 0
    phase1_success_blocks: int = 0
    phase1_retry_blocks: int = 0
    phase1_failed_blocks: int = 0
    repair_invocations: int = 0
    repair_accepted: int = 0
    repair_rejected: int = 0
    style_retry_invocations: int = 0
    style_retry_accepted: int = 0
    style_retry_rejected: int = 0
    local_rewrap_attempts: int = 0
    local_rewrap_successes: int = 0
    smaller_block_fallbacks: int = 0
    single_cue_source_fallbacks: int = 0
    single_cue_issue_keeps: int = 0
    failure_reasons: Dict[str, int] = field(default_factory=dict)
    pre_wrap_failures: Dict[str, int] = field(default_factory=dict)
    post_wrap_failures: Dict[str, int] = field(default_factory=dict)
    phase1_risk_flags: Dict[str, int] = field(default_factory=dict)
    strict_retry_candidate_risk_flags: Dict[str, int] = field(default_factory=dict)
    style_retry_rejection_causes: Dict[str, int] = field(default_factory=dict)
    style_action_attempts: Dict[str, int] = field(default_factory=dict)
    style_action_accepts: Dict[str, int] = field(default_factory=dict)
    style_action_rejections: Dict[str, int] = field(default_factory=dict)
    style_action_remaining_warnings: Dict[str, int] = field(default_factory=dict)
    style_action_tail_attempts: Dict[str, int] = field(default_factory=dict)
    style_action_tail_accepts: Dict[str, int] = field(default_factory=dict)
    style_action_tail_rejections: Dict[str, int] = field(default_factory=dict)
    style_retry_trace: Dict[str, Any] = field(default_factory=dict)
    glossary_hard_violations: int = 0
    front_sparse_count: int = 0
    tail_heavy_count: int = 0
    post_wrap_failure_blocks: int = 0
    final_cps_sum: float = 0.0
    final_cue_count: int = 0

    def add_reasons(self, bucket: str, reasons: Iterable[str]) -> None:
        target = {
            "failure": self.failure_reasons,
            "pre_wrap": self.pre_wrap_failures,
            "post_wrap": self.post_wrap_failures,
        }[bucket]
        for reason in reasons:
            target[reason] = target.get(reason, 0) + 1
            if bucket == "failure":
                if reason == "glossary_violation":
                    self.glossary_hard_violations += 1
                elif reason == "front_sparse":
                    self.front_sparse_count += 1
                elif reason == "tail_heavy":
                    self.tail_heavy_count += 1

    def note_final_cues(self, cues: Iterable[Cue]) -> None:
        for cue in cues:
            duration_ms = max(ts_to_ms(cue.end) - ts_to_ms(cue.start), 1)
            visible_chars = len(normalize_text(cue.text).replace(" ", ""))
            self.final_cps_sum += visible_chars / (duration_ms / 1000.0)
            self.final_cue_count += 1

    def add_phase1_risk_flags(self, risk_flags: Iterable[str]) -> None:
        for flag in risk_flags:
            self.phase1_risk_flags[flag] = self.phase1_risk_flags.get(flag, 0) + 1

    def add_strict_retry_candidate_risk_flags(self, risk_flags: Iterable[str]) -> None:
        for flag in risk_flags:
            self.strict_retry_candidate_risk_flags[flag] = self.strict_retry_candidate_risk_flags.get(flag, 0) + 1

    def add_style_retry_rejection_causes(self, causes: Iterable[str]) -> None:
        for cause in causes:
            self.style_retry_rejection_causes[cause] = self.style_retry_rejection_causes.get(cause, 0) + 1

    def note_style_action_attempts(self, spans: Iterable[Dict[str, Any]]) -> None:
        for span in spans:
            action = span.get("preferred_action")
            if not action:
                continue
            self.style_action_attempts[action] = self.style_action_attempts.get(action, 0) + 1
            tail_type = span.get("source_tail_type")
            if tail_type:
                key = f"{action}|{tail_type}"
                self.style_action_tail_attempts[key] = self.style_action_tail_attempts.get(key, 0) + 1

    def note_style_action_outcome(self, spans: Iterable[Dict[str, Any]], accepted: bool) -> None:
        for span in spans:
            action = span.get("preferred_action")
            if not action:
                continue
            target = self.style_action_accepts if accepted else self.style_action_rejections
            target[action] = target.get(action, 0) + 1
            tail_type = span.get("source_tail_type")
            if tail_type:
                keyed = f"{action}|{tail_type}"
                tail_target = self.style_action_tail_accepts if accepted else self.style_action_tail_rejections
                tail_target[keyed] = tail_target.get(keyed, 0) + 1

    def note_style_action_remaining_warnings(self, spans: Iterable[Dict[str, Any]]) -> None:
        for span in spans:
            action = span.get("preferred_action")
            if not action:
                continue
            self.style_action_remaining_warnings[action] = self.style_action_remaining_warnings.get(action, 0) + 1

    def average_cps(self) -> float:
        if self.final_cue_count == 0:
            return 0.0
        return self.final_cps_sum / self.final_cue_count

    def to_payload(self, input_path: str, output_path: str, phase1_model: str, repair_model: str) -> dict:
        blocks = self.blocks_started or 1
        repairs = self.repair_invocations or 1
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_path": input_path,
            "output_path": output_path,
            "phase1_model": phase1_model,
            "repair_model": repair_model,
            "blocks_started": self.blocks_started,
            "phase1_success_rate": self.phase1_success_blocks / blocks,
            "phase1_retry_rate": self.phase1_retry_blocks / blocks,
            "repair_invocation_rate": self.repair_invocations / blocks,
            "repair_success_rate": self.repair_accepted / repairs,
            "style_retry_invocation_rate": self.style_retry_invocations / blocks,
            "style_retry_success_rate": self.style_retry_accepted / max(self.style_retry_invocations, 1),
            "smaller_block_fallback_rate": self.smaller_block_fallbacks / blocks,
            "single_cue_fallback_rate": self.single_cue_source_fallbacks / blocks,
            "post_wrap_failure_rate": self.post_wrap_failure_blocks / blocks,
            "glossary_hard_violations": self.glossary_hard_violations,
            "front_sparse_count": self.front_sparse_count,
            "tail_heavy_count": self.tail_heavy_count,
            "average_cps": self.average_cps(),
            "failure_reasons": self.failure_reasons,
            "pre_wrap_failures": self.pre_wrap_failures,
            "post_wrap_failures": self.post_wrap_failures,
            "phase1_risk_flags": self.phase1_risk_flags,
            "strict_retry_candidate_risk_flags": self.strict_retry_candidate_risk_flags,
            "style_retry_rejection_causes": self.style_retry_rejection_causes,
            "style_action_attempts": self.style_action_attempts,
            "style_action_accepts": self.style_action_accepts,
            "style_action_rejections": self.style_action_rejections,
            "style_action_remaining_warnings": self.style_action_remaining_warnings,
            "style_action_tail_attempts": self.style_action_tail_attempts,
            "style_action_tail_accepts": self.style_action_tail_accepts,
            "style_action_tail_rejections": self.style_action_tail_rejections,
        }

    def summary(self) -> str:
        blocks = self.blocks_started or 1
        return (
            f"blocks={self.blocks_started} phase1_retry_rate={self.phase1_retry_blocks / blocks:.2%} "
            f"repair_rate={self.repair_invocations / blocks:.2%} repair_success={self.repair_accepted}/{self.repair_invocations} "
            f"style_retry_success={self.style_retry_accepted}/{self.style_retry_invocations} "
            f"fallback_rate={self.smaller_block_fallbacks / blocks:.2%} avg_cps={self.average_cps():.2f}"
        )


def append_metrics_log(path: Optional[Path], metrics: TranslationMetrics, input_path: str, output_path: str, phase1_model: str, repair_model: str) -> None:
    if not path:
        return
    payload = metrics.to_payload(
        input_path=input_path,
        output_path=output_path,
        phase1_model=phase1_model,
        repair_model=repair_model,
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:
        warn(f"Failed to write metrics log {path}: {exc}")
