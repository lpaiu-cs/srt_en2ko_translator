from __future__ import annotations

import json
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import re

import requests

from .config import RuntimeConfig
from .models import EmittedCue, PhaseTranslationResult, RepairRequest, TranslationRequest
from .splitting import ts_to_ms
from .text import compact_spaces, normalize_text

RISK_FLAG_ENUM = [
    "unsupported_head_marker_risk",
    "unsupported_explanatory_tail_risk",
    "duplicate_restatement_risk",
    "english_residual_risk",
    "line_readability_risk",
    "glossary_uncertain",
    "context_uncertain",
]

_REPAIR_TECHNICAL_HINT_RE = re.compile(r"[A-Za-z][A-Za-z0-9./+-]{2,}")
_REPAIR_TECHNICAL_HINTS = {
    "alexnet",
    "c3d",
    "convnet",
    "dc-gan",
    "gflops",
    "gpt-4",
    "rmsprop",
    "sgd",
    "vgg-16",
}


def _extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") in {"text", "output_text"} and item.get("text"):
                parts.append(str(item["text"]))
        return "".join(parts).strip()
    raise ValueError("Unsupported message content format")


def _build_phase_schema(cue_count: int, schema_name: str) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "emitted_cues": {
                        "type": "array",
                        "minItems": cue_count,
                        "maxItems": cue_count,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "cue_index": {"type": "integer"},
                                "text": {"type": "string"},
                            },
                            "required": ["cue_index", "text"],
                        },
                    },
                    "risk_flags": {
                        "type": "array",
                        "items": {"type": "string", "enum": RISK_FLAG_ENUM},
                    },
                },
                "required": ["emitted_cues", "risk_flags"],
            },
        },
    }


def _build_offending_retry_schema(cue_count: int, schema_name: str) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "offending_cue_rewrites": {
                        "type": "array",
                        "minItems": cue_count,
                        "maxItems": cue_count,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "cue_index": {"type": "integer"},
                                "text": {"type": "string"},
                            },
                            "required": ["cue_index", "text"],
                        },
                    },
                    "risk_flags": {
                        "type": "array",
                        "items": {"type": "string", "enum": RISK_FLAG_ENUM},
                    },
                },
                "required": ["offending_cue_rewrites", "risk_flags"],
            },
        },
    }


def _splice_offending_cue_rewrites(
    request: TranslationRequest,
    rewrites: List[dict],
    risk_flags: List[str],
) -> PhaseTranslationResult:
    rewrite_map = {
        int(item["cue_index"]): normalize_text(str(item["text"]))
        for item in rewrites
    }
    previous_map = {
        cue.cue_index: normalize_text(cue.text)
        for cue in request.previous_emitted_cues
    }
    emitted_cues: List[EmittedCue] = []
    for cue in request.block.cues:
        if cue.index in rewrite_map:
            emitted_cues.append(EmittedCue(cue_index=cue.index, text=rewrite_map[cue.index]))
            continue
        if cue.index not in previous_map:
            raise ValueError(f"Missing protected cue {cue.index} in previous_emitted_cues for offending-only retry")
        emitted_cues.append(EmittedCue(cue_index=cue.index, text=previous_map[cue.index]))
    return PhaseTranslationResult(
        emitted_cues=emitted_cues,
        risk_flags=list(dict.fromkeys(risk_flags)),
    )


def _phase1_examples_v1() -> List[tuple[dict, dict]]:
    return [
        (
            {
                "task": "subtitle_phase1_translate",
                "boundary_lint": {
                    "lint_flags": ["dependent_end"],
                    "lint_actions": ["carry_context_only"],
                },
                "source_cues": [
                    {
                        "cue_index": 101,
                        "start": "00:00:01,000",
                        "end": "00:00:03,000",
                        "text": "And the question was, if we use the same across the networks,",
                        "duration_ms": 2000,
                        "gap_after_ms": 0,
                    }
                ],
                "left_context": [],
                "right_context": [],
                "glossary_terms": [],
            },
            {
                "emitted_cues": [
                    {"cue_index": 101, "text": "그리고 질문은, 네트워크 전반에 걸쳐 같은 것을 사용한다면,"}
                ],
                "risk_flags": [],
            },
        ),
        (
            {
                "task": "subtitle_phase1_translate",
                "boundary_lint": {
                    "lint_flags": ["numeric_orphan"],
                    "lint_actions": ["carry_context_only"],
                },
                "source_cues": [
                    {
                        "cue_index": 201,
                        "start": "00:00:04,000",
                        "end": "00:00:05,000",
                        "text": "of 13% or 0.13.",
                        "duration_ms": 1000,
                        "gap_after_ms": 0,
                    }
                ],
                "left_context": [],
                "right_context": [],
                "glossary_terms": [],
            },
            {
                "emitted_cues": [
                    {"cue_index": 201, "text": "13% 또는 0.13"}
                ],
                "risk_flags": [],
            },
        ),
        (
            {
                "task": "subtitle_phase1_translate",
                "boundary_lint": {
                    "lint_flags": ["qa_fragment"],
                    "lint_actions": ["carry_context_only"],
                },
                "source_cues": [
                    {
                        "cue_index": 301,
                        "start": "00:00:06,000",
                        "end": "00:00:07,000",
                        "text": "Yeah, yeah.",
                        "duration_ms": 1000,
                        "gap_after_ms": 0,
                    }
                ],
                "left_context": [],
                "right_context": [],
                "glossary_terms": [],
            },
            {
                "emitted_cues": [
                    {"cue_index": 301, "text": "네, 네."}
                ],
                "risk_flags": [],
            },
        ),
        (
            {
                "task": "subtitle_phase1_translate",
                "boundary_lint": {
                    "lint_flags": [],
                    "lint_actions": [],
                },
                "source_cues": [
                    {
                        "cue_index": 401,
                        "start": "00:00:08,000",
                        "end": "00:00:11,000",
                        "text": "It's very low loss when you're predicting the correct class at very high probability.",
                        "duration_ms": 3000,
                        "gap_after_ms": 0,
                    }
                ],
                "left_context": [],
                "right_context": [],
                "glossary_terms": [],
            },
            {
                "emitted_cues": [
                    {"cue_index": 401, "text": "정답 클래스를 매우 높은 확률로 예측할 때 손실은 매우 낮습니다."}
                ],
                "risk_flags": [],
            },
        ),
        (
            {
                "task": "subtitle_phase1_translate",
                "boundary_lint": {
                    "lint_flags": ["dependent_end"],
                    "lint_actions": ["carry_context_only"],
                },
                "style_retry": {
                    "strict_retry": False,
                    "retry_reasons": [],
                    "bad_output_to_avoid": "여기서 흔히 볼 수 있는 주황색 블록들은, 3x3 컨볼루션 레이어입니다. 즉, 이런 것들이 컨볼루션 레이어라는 겁니다,",
                    "note": "Do not inject discourse markers like 즉 or 다시 말해 to close a relative-clause tail.",
                },
                "source_cues": [
                    {
                        "cue_index": 481,
                        "start": "00:00:15,500",
                        "end": "00:00:17,300",
                        "text": "So these orange blocks",
                        "duration_ms": 1800,
                        "gap_after_ms": 40,
                    },
                    {
                        "cue_index": 482,
                        "start": "00:00:17,340",
                        "end": "00:00:18,800",
                        "text": "are 3 by 3 convolution layers.",
                        "duration_ms": 1460,
                        "gap_after_ms": 50,
                    },
                    {
                        "cue_index": 483,
                        "start": "00:00:18,850",
                        "end": "00:00:20,100",
                        "text": "So these are convolution layers that",
                        "duration_ms": 1250,
                        "gap_after_ms": 0,
                    },
                ],
                "left_context": [],
                "right_context": [],
                "glossary_terms": [],
            },
            {
                "emitted_cues": [
                    {"cue_index": 481, "text": "여기서 흔히 볼 수 있는 주황색 블록들은,"},
                    {"cue_index": 482, "text": "3x3 컨볼루션 레이어입니다."},
                    {"cue_index": 483, "text": "이런 건 컨볼루션 레이어인데"},
                ],
                "risk_flags": [],
            },
        ),
    ]


def _phase1_examples_v2() -> List[tuple[dict, dict]]:
    return [
        (
            {
                "task": "subtitle_phase1_translate",
                "boundary_lint": {
                    "lint_flags": ["dependent_end"],
                    "lint_actions": ["carry_context_only"],
                },
                "style_retry": {
                    "strict_retry": False,
                    "retry_reasons": [],
                    "bad_output_to_avoid": "그리고 그 학습이 끝나면, 같은 2단계 파이프라인을 따른다는 겁니다.",
                    "note": "Do not over-close an unfinished fragment with a lecture-style explanation.",
                },
                "source_cues": [
                    {
                        "cue_index": 101,
                        "start": "00:00:01,000",
                        "end": "00:00:02,400",
                        "text": "And then once they were done training that,",
                        "duration_ms": 1400,
                        "gap_after_ms": 60,
                    },
                    {
                        "cue_index": 102,
                        "start": "00:00:02,460",
                        "end": "00:00:04,200",
                        "text": "you follow the same 2-step pipeline that you saw",
                        "duration_ms": 1740,
                        "gap_after_ms": 0,
                    },
                ],
                "left_context": [],
                "right_context": [],
                "glossary_terms": [],
            },
            {
                "emitted_cues": [
                    {"cue_index": 101, "text": "그리고 그 학습이 끝나면,"},
                    {"cue_index": 102, "text": "여러분이 봤던 같은 2단계 파이프라인을 따르고,"},
                ],
                "risk_flags": [],
            },
        ),
        (
            {
                "task": "subtitle_phase1_translate",
                "boundary_lint": {
                    "lint_flags": ["numeric_orphan"],
                    "lint_actions": ["carry_context_only"],
                },
                "style_retry": {
                    "strict_retry": False,
                    "retry_reasons": [],
                    "bad_output_to_avoid": "13% 또는 0.13입니다.",
                    "note": "Preserve the numeric fragment without adding a sentence-final copula.",
                },
                "source_cues": [
                    {
                        "cue_index": 201,
                        "start": "00:00:04,000",
                        "end": "00:00:05,000",
                        "text": "of 13% or 0.13.",
                        "duration_ms": 1000,
                        "gap_after_ms": 0,
                    }
                ],
                "left_context": [],
                "right_context": [],
                "glossary_terms": [],
            },
            {
                "emitted_cues": [
                    {"cue_index": 201, "text": "13% 또는 0.13"}
                ],
                "risk_flags": [],
            },
        ),
        (
            {
                "task": "subtitle_phase1_translate",
                "boundary_lint": {
                    "lint_flags": ["comparison_midstart"],
                    "lint_actions": ["carry_context_only"],
                },
                "style_retry": {
                    "strict_retry": False,
                    "retry_reasons": [],
                    "bad_output_to_avoid": "이는 d_k의 제곱근으로 나눈다는 뜻입니다.",
                    "note": "A comparison fragment should not invent a full explanatory lead-in.",
                },
                "source_cues": [
                    {
                        "cue_index": 251,
                        "start": "00:00:05,200",
                        "end": "00:00:06,100",
                        "text": "like dividing by",
                        "duration_ms": 900,
                        "gap_after_ms": 40,
                    },
                    {
                        "cue_index": 252,
                        "start": "00:00:06,140",
                        "end": "00:00:07,400",
                        "text": "the square root of d_k.",
                        "duration_ms": 1260,
                        "gap_after_ms": 0,
                    },
                ],
                "left_context": [],
                "right_context": [],
                "glossary_terms": [],
            },
            {
                "emitted_cues": [
                    {"cue_index": 251, "text": "예를 들어"},
                    {"cue_index": 252, "text": "d_k의 제곱근으로 나누는 것처럼요."},
                ],
                "risk_flags": [],
            },
        ),
        (
            {
                "task": "subtitle_phase1_translate",
                "boundary_lint": {
                    "lint_flags": ["qa_fragment"],
                    "lint_actions": ["carry_context_only"],
                },
                "style_retry": {
                    "strict_retry": False,
                    "retry_reasons": [],
                    "bad_output_to_avoid": "네, 맞습니다. 그런 뜻입니다.",
                    "note": "Keep acknowledgements short instead of expanding them.",
                },
                "source_cues": [
                    {
                        "cue_index": 301,
                        "start": "00:00:06,000",
                        "end": "00:00:07,000",
                        "text": "Yeah, yeah.",
                        "duration_ms": 1000,
                        "gap_after_ms": 0,
                    }
                ],
                "left_context": [],
                "right_context": [],
                "glossary_terms": [],
            },
            {
                "emitted_cues": [
                    {"cue_index": 301, "text": "네, 네."}
                ],
                "risk_flags": [],
            },
        ),
        (
            {
                "task": "subtitle_phase1_translate",
                "boundary_lint": {
                    "lint_flags": [],
                    "lint_actions": [],
                },
                "style_retry": {
                    "strict_retry": False,
                    "retry_reasons": [],
                    "bad_output_to_avoid": "하지만 실제로 진전을 이루려면 여기서는 조금 더 구조가 필요합니다. 조금 더 구조가 필요합니다.",
                    "note": "Do not repeat the same proposition in a second paraphrase.",
                },
                "source_cues": [
                    {
                        "cue_index": 401,
                        "start": "00:00:08,600",
                        "end": "00:00:11,400",
                        "text": "But to make progress here, we need a bit more structure.",
                        "duration_ms": 2800,
                        "gap_after_ms": 0,
                    }
                ],
                "left_context": [],
                "right_context": [],
                "glossary_terms": [],
            },
            {
                "emitted_cues": [
                    {"cue_index": 401, "text": "하지만 여기서 진전을 이루려면 조금 더 구조가 필요합니다."}
                ],
                "risk_flags": [],
            },
        ),
        (
            {
                "task": "subtitle_phase1_translate",
                "boundary_lint": {
                    "lint_flags": [],
                    "lint_actions": [],
                },
                "style_retry": {
                    "strict_retry": False,
                    "retry_reasons": [],
                    "bad_output_to_avoid": "그리고 다시 이미지 해상도로 돌아가는 업샘플링 단계가 있습니다. 이미지 해상도로 돌아가는 단계입니다.",
                    "note": "Keep one concise proposition instead of repeating the tail.",
                },
                "source_cues": [
                    {
                        "cue_index": 451,
                        "start": "00:00:11,500",
                        "end": "00:00:13,500",
                        "text": "and then upsampling phase",
                        "duration_ms": 2000,
                        "gap_after_ms": 50,
                    },
                    {
                        "cue_index": 452,
                        "start": "00:00:13,550",
                        "end": "00:00:15,400",
                        "text": "that goes back to the image resolution.",
                        "duration_ms": 1850,
                        "gap_after_ms": 0,
                    },
                ],
                "left_context": [],
                "right_context": [],
                "glossary_terms": [],
            },
            {
                "emitted_cues": [
                    {"cue_index": 451, "text": "그리고 다시 이미지 해상도로 돌아가는"},
                    {"cue_index": 452, "text": "업샘플링 단계가 있습니다."},
                ],
                "risk_flags": [],
            },
        ),
    ]


def _phase1_example_messages(
    prompt_profile: str,
    strict_style_retry: bool = False,
    strict_retry_mode: str = "full_block",
) -> List[dict]:
    if strict_style_retry and strict_retry_mode == "offending_cue_only" and prompt_profile == "fragment_preserving_v2":
        offending_only_user = {
            "task": "subtitle_phase1_translate",
            "boundary_lint": {
                "lint_flags": ["dependent_start", "comparison_midstart"],
                "lint_actions": ["carry_context_only"],
            },
            "style_retry": {
                "strict_retry": True,
                "retry_contract": "offending_cue_only",
                "retry_reasons": ["duplicate_restatement"],
                "bad_output_to_avoid": "모델이 할 수 있는 것 중 하나는 이미지와 텍스트의 연관성만으로 학습할 수 있다는 점입니다. 이미지와 텍스트의 연관성만으로 학습할 수 있다는 점입니다.",
                "note": "Rewrite only cue 192. Keep cue 191 verbatim and return only the offending cue rewrite.",
            },
            "source_cues": [
                {
                    "cue_index": 191,
                    "start": "00:07:58,470",
                    "end": "00:08:02,070",
                    "text": "like model is that it can be trained with just associations",
                    "duration_ms": 3600,
                    "gap_after_ms": 0,
                },
                {
                    "cue_index": 192,
                    "start": "00:08:02,070",
                    "end": "00:08:03,650",
                    "text": "of images and text.",
                    "duration_ms": 1580,
                    "gap_after_ms": 0,
                },
            ],
            "left_context": [],
            "right_context": [],
            "glossary_terms": [],
            "previous_emitted_cues": [
                {"cue_index": 191, "text": "모델이 할 수 있는 것 중 하나는 이미지와 텍스트의 연관성만으로 학습할 수 있다는 점입니다."},
                {"cue_index": 192, "text": "이미지와 텍스트의 연관성만으로 학습할 수 있다는 점입니다."},
            ],
            "offending_cue_indices": [192],
            "protected_cue_indices": [191],
            "preferred_actions": ["restore_missing_tail"],
            "offending_spans": [
                {
                    "cue_index": 192,
                    "cue_indices": [191, 192],
                    "span_text": "이미지와 텍스트의 연관성만으로 학습할 수 있다는 점입니다",
                    "left_text": "모델이 할 수 있는 것 중 하나는 이미지와 텍스트의 연관성만으로 학습할 수 있다는 점입니다",
                    "right_text": "이미지와 텍스트의 연관성만으로 학습할 수 있다는 점입니다",
                    "source_cue_text": "of images and text.",
                    "source_tail_type": "continuation_tail",
                    "issue": "duplicate_restatement",
                    "preferred_action": "restore_missing_tail",
                    "trigger_reason": "detector_miss",
                }
            ],
        }
        offending_only_assistant = {
            "offending_cue_rewrites": [
                {"cue_index": 192, "text": "이미지와 텍스트의 연관성으로요."}
            ],
            "risk_flags": [],
        }
        return [
            {"role": "user", "content": json.dumps(offending_only_user, ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps(offending_only_assistant, ensure_ascii=False)},
        ]

    if prompt_profile == "fragment_preserving_v1":
        examples = _phase1_examples_v1()
    elif prompt_profile in {"fragment_preserving_v2", "fragment_preserving_v3"}:
        examples = _phase1_examples_v2()
    else:
        return []
    messages: List[dict] = []
    for user_payload, assistant_payload in examples:
        messages.append({"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)})
        messages.append({"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=False)})
    if strict_style_retry and prompt_profile in {"fragment_preserving_v2", "fragment_preserving_v3"}:
        strict_user = {
            "task": "subtitle_phase1_translate",
            "boundary_lint": {
                "lint_flags": ["dependent_end"],
                "lint_actions": ["carry_context_only"],
            },
            "style_retry": {
                "strict_retry": True,
                "retry_reasons": ["unsupported_explanatory_tail", "duplicate_restatement"],
                "bad_output_to_avoid": "그리고 그 학습이 끝나면, 여러분이 봤던 같은 2단계 파이프라인을 따른다는 겁니다. 같은 파이프라인을 따르게 되는 거죠.",
                "note": "On a strict retry, remove both the explanatory closure and the repeated paraphrase.",
            },
            "source_cues": [
                {
                    "cue_index": 501,
                    "start": "00:00:15,500",
                    "end": "00:00:16,600",
                    "text": "And then once they were done training that,",
                    "duration_ms": 1100,
                    "gap_after_ms": 40,
                },
                {
                    "cue_index": 502,
                    "start": "00:00:16,640",
                    "end": "00:00:18,300",
                    "text": "you follow the same 2-step pipeline that you saw",
                    "duration_ms": 1660,
                    "gap_after_ms": 0,
                },
            ],
            "left_context": [],
            "right_context": [],
            "glossary_terms": [],
        }
        strict_assistant = {
            "emitted_cues": [
                {"cue_index": 501, "text": "그리고 그 학습이 끝나면,"},
                {"cue_index": 502, "text": "여러분이 봤던 같은 2단계 파이프라인을 따르고,"},
            ],
            "risk_flags": [],
        }
        messages.append({"role": "user", "content": json.dumps(strict_user, ensure_ascii=False)})
        messages.append({"role": "assistant", "content": json.dumps(strict_assistant, ensure_ascii=False)})
        strict_head_user = {
            "task": "subtitle_phase1_translate",
            "boundary_lint": {
                "lint_flags": ["dependent_end"],
                "lint_actions": ["carry_context_only"],
            },
            "style_retry": {
                "strict_retry": True,
                "retry_reasons": ["unsupported_head_marker"],
                "bad_output_to_avoid": "즉, 이런 컨볼루션 레이어들은",
                "note": "Delete the unsupported discourse marker head and keep the fragment open.",
            },
            "source_cues": [
                {
                    "cue_index": 541,
                    "start": "00:00:19,000",
                    "end": "00:00:20,100",
                    "text": "So these are convolution layers that",
                    "duration_ms": 1100,
                    "gap_after_ms": 0,
                }
            ],
            "left_context": [],
            "right_context": [],
            "glossary_terms": [],
        }
        strict_head_assistant = {
            "emitted_cues": [
                {"cue_index": 541, "text": "이런 컨볼루션 레이어들은"}
            ],
            "risk_flags": [],
        }
        messages.append({"role": "user", "content": json.dumps(strict_head_user, ensure_ascii=False)})
        messages.append({"role": "assistant", "content": json.dumps(strict_head_assistant, ensure_ascii=False)})
        strict_purpose_user = {
            "task": "subtitle_phase1_translate",
            "boundary_lint": {
                "lint_flags": [],
                "lint_actions": [],
            },
            "style_retry": {
                "strict_retry": True,
                "retry_reasons": ["duplicate_restatement"],
                "bad_output_to_avoid": "하지만 실제로 진전을 이루려면 여기서 조금 더 구조가 필요합니다. 조금 더 구조가 필요합니다.",
                "note": "Restore the missing purpose-tail meaning in cue 2 without repeating cue 1.",
            },
            "source_cues": [
                {
                    "cue_index": 1020,
                    "start": "00:00:40,000",
                    "end": "00:00:41,800",
                    "text": "But we need a little bit more structure here",
                    "duration_ms": 1800,
                    "gap_after_ms": 50,
                },
                {
                    "cue_index": 1021,
                    "start": "00:00:41,850",
                    "end": "00:00:43,900",
                    "text": "to actually make progress.",
                    "duration_ms": 2050,
                    "gap_after_ms": 0,
                },
            ],
            "left_context": [],
            "right_context": [],
            "glossary_terms": [],
            "previous_emitted_cues": [
                {"cue_index": 1020, "text": "하지만 실제로 진전을 이루려면 여기서 조금 더 구조가 필요합니다."},
                {"cue_index": 1021, "text": "조금 더 구조가 필요합니다."},
            ],
            "offending_cue_indices": [1021],
            "protected_cue_indices": [1020],
            "preferred_actions": ["restore_missing_tail"],
            "offending_spans": [
                {
                    "cue_index": 1021,
                    "cue_indices": [1020, 1021],
                    "span_text": "조금 더 구조가 필요합니다.",
                    "left_text": "하지만 실제로 진전을 이루려면 여기서 조금 더 구조가 필요합니다.",
                    "right_text": "조금 더 구조가 필요합니다.",
                    "source_cue_text": "to actually make progress.",
                    "source_tail_type": "purpose_tail",
                    "issue": "duplicate_restatement",
                    "preferred_action": "restore_missing_tail",
                }
            ],
        }
        strict_purpose_assistant = {
            "emitted_cues": [
                {"cue_index": 1020, "text": "하지만 실제로 진전을 이루려면 여기서 조금 더 구조가 필요합니다."},
                {"cue_index": 1021, "text": "실제로 진전을 이루기 위해서요."},
            ],
            "risk_flags": [],
        }
        messages.append({"role": "user", "content": json.dumps(strict_purpose_user, ensure_ascii=False)})
        messages.append({"role": "assistant", "content": json.dumps(strict_purpose_assistant, ensure_ascii=False)})
        strict_relative_user = {
            "task": "subtitle_phase1_translate",
            "boundary_lint": {
                "lint_flags": [],
                "lint_actions": [],
            },
            "style_retry": {
                "strict_retry": True,
                "retry_reasons": ["duplicate_restatement"],
                "bad_output_to_avoid": "50이 꽤 좋은 값입니다. 대부분의 사람들이 실제로 사용합니다.",
                "note": "Restore the relative-clause tail as a modifier, not as a new closed sentence.",
            },
            "source_cues": [
                {
                    "cue_index": 1198,
                    "start": "00:01:10,000",
                    "end": "00:01:12,000",
                    "text": "but 50 is a pretty good one",
                    "duration_ms": 2000,
                    "gap_after_ms": 30,
                },
                {
                    "cue_index": 1199,
                    "start": "00:01:12,030",
                    "end": "00:01:13,400",
                    "text": "that most people use in practice.",
                    "duration_ms": 1370,
                    "gap_after_ms": 0,
                },
            ],
            "left_context": [],
            "right_context": [],
            "glossary_terms": [],
            "previous_emitted_cues": [
                {"cue_index": 1198, "text": "하지만 50이 꽤 좋은 값입니다."},
                {"cue_index": 1199, "text": "대부분의 사람들이 실제로 사용합니다."},
            ],
            "offending_cue_indices": [1199],
            "protected_cue_indices": [1198],
            "preferred_actions": ["restore_missing_tail"],
            "offending_spans": [
                {
                    "cue_index": 1199,
                    "cue_indices": [1198, 1199],
                    "span_text": "대부분의 사람들이 실제로 사용합니다.",
                    "left_text": "하지만 50이 꽤 좋은 값입니다.",
                    "right_text": "대부분의 사람들이 실제로 사용합니다.",
                    "source_cue_text": "that most people use in practice.",
                    "source_tail_type": "relative_clause_tail",
                    "issue": "duplicate_restatement",
                    "preferred_action": "restore_missing_tail",
                }
            ],
        }
        strict_relative_assistant = {
            "emitted_cues": [
                {"cue_index": 1198, "text": "하지만 50이 꽤 좋은 값이고,"},
                {"cue_index": 1199, "text": "대부분의 사람들이 실제로 사용하는"},
            ],
            "risk_flags": [],
        }
        messages.append({"role": "user", "content": json.dumps(strict_relative_user, ensure_ascii=False)})
        messages.append({"role": "assistant", "content": json.dumps(strict_relative_assistant, ensure_ascii=False)})
        strict_continuation_user = {
            "task": "subtitle_phase1_translate",
            "boundary_lint": {
                "lint_flags": [],
                "lint_actions": [],
            },
            "style_retry": {
                "strict_retry": True,
                "retry_reasons": ["duplicate_restatement"],
                "bad_output_to_avoid": "이상적으로는 CLIP 모델을 바로 바로 사용할 수 있기를 원합니다.",
                "note": "When the first cue already carries the setup, restore only the leftover continuation tail instead of restating the whole proposition as a full sentence. Do not repeat the same adverbial anchor from cue 1 at the start of cue 2.",
            },
            "source_cues": [
                {
                    "cue_index": 1240,
                    "start": "00:01:20,000",
                    "end": "00:01:21,800",
                    "text": "We would ideally want to be",
                    "duration_ms": 1800,
                    "gap_after_ms": 40,
                },
                {
                    "cue_index": 1241,
                    "start": "00:01:21,840",
                    "end": "00:01:23,300",
                    "text": "able to use a CLIP model out of the box.",
                    "duration_ms": 1460,
                    "gap_after_ms": 0,
                },
            ],
            "left_context": [],
            "right_context": [],
            "glossary_terms": [],
            "previous_emitted_cues": [
                {"cue_index": 1240, "text": "이상적으로는 CLIP 모델을 바로"},
                {"cue_index": 1241, "text": "바로 사용할 수 있기를 원합니다."},
            ],
            "offending_cue_indices": [1241],
            "protected_cue_indices": [1240],
            "preferred_actions": ["restore_missing_tail"],
            "offending_spans": [
                {
                    "cue_index": 1241,
                    "cue_indices": [1240, 1241],
                    "span_text": "바로 사용할 수 있기를 원합니다.",
                    "left_text": "이상적으로는 CLIP 모델을 바로",
                    "right_text": "바로 사용할 수 있기를 원합니다.",
                    "source_cue_text": "able to use a CLIP model out of the box.",
                    "source_tail_type": "continuation_tail",
                    "issue": "duplicate_restatement",
                    "preferred_action": "restore_missing_tail",
                }
            ],
        }
        strict_continuation_assistant = {
            "emitted_cues": [
                {"cue_index": 1240, "text": "이상적으로는 CLIP 모델을 바로"},
                {"cue_index": 1241, "text": "즉시 사용할 수 있으면 좋겠죠."},
            ],
            "risk_flags": [],
        }
        messages.append({"role": "user", "content": json.dumps(strict_continuation_user, ensure_ascii=False)})
        messages.append({"role": "assistant", "content": json.dumps(strict_continuation_assistant, ensure_ascii=False)})
        strict_continuation_short_tail_user = {
            "task": "subtitle_phase1_translate",
            "boundary_lint": {
                "lint_flags": [],
                "lint_actions": [],
            },
            "style_retry": {
                "strict_retry": True,
                "retry_reasons": ["duplicate_restatement"],
                "bad_output_to_avoid": "모델의 장점은 이미지와 텍스트의 연관성만으로도 학습할 수 있다는 점이고, 이미지와 텍스트의 연관성만으로 학습할 수 있다는 점입니다.",
                "note": "If the protected cue already contains the main predicate, keep only the missing continuation tail in the offending cue.",
            },
            "source_cues": [
                {
                    "cue_index": 1290,
                    "start": "00:01:24,000",
                    "end": "00:01:25,900",
                    "text": "one nice thing about the model is that it can be trained with just associations",
                    "duration_ms": 1900,
                    "gap_after_ms": 40,
                },
                {
                    "cue_index": 1291,
                    "start": "00:01:25,940",
                    "end": "00:01:27,300",
                    "text": "of images and text.",
                    "duration_ms": 1360,
                    "gap_after_ms": 0,
                },
            ],
            "left_context": [],
            "right_context": [],
            "glossary_terms": [],
            "previous_emitted_cues": [
                {"cue_index": 1290, "text": "모델의 장점은 이미지와 텍스트의 연관성만으로도 학습할 수 있다는 점이고,"},
                {"cue_index": 1291, "text": "이미지와 텍스트의 연관성만으로 학습할 수 있다는 점입니다."},
            ],
            "offending_cue_indices": [1291],
            "protected_cue_indices": [1290],
            "preferred_actions": ["restore_missing_tail"],
            "offending_spans": [
                {
                    "cue_index": 1291,
                    "cue_indices": [1290, 1291],
                    "span_text": "이미지와 텍스트의 연관성만으로 학습할 수 있다는 점입니다.",
                    "left_text": "모델의 장점은 이미지와 텍스트의 연관성만으로도 학습할 수 있다는 점이고,",
                    "right_text": "이미지와 텍스트의 연관성만으로 학습할 수 있다는 점입니다.",
                    "source_cue_text": "of images and text.",
                    "source_tail_type": "continuation_tail",
                    "issue": "duplicate_restatement",
                    "preferred_action": "restore_missing_tail",
                }
            ],
        }
        strict_continuation_short_tail_assistant = {
            "emitted_cues": [
                {"cue_index": 1290, "text": "모델의 장점은 이미지와 텍스트의 연관성만으로도 학습할 수 있다는 점이고,"},
                {"cue_index": 1291, "text": "이미지와 텍스트의 연관성만으로요."},
            ],
            "risk_flags": [],
        }
        messages.append({"role": "user", "content": json.dumps(strict_continuation_short_tail_user, ensure_ascii=False)})
        messages.append({"role": "assistant", "content": json.dumps(strict_continuation_short_tail_assistant, ensure_ascii=False)})
        strict_continuation_detector_miss_user = {
            "task": "subtitle_phase1_translate",
            "boundary_lint": {
                "lint_flags": ["dependent_start", "comparison_midstart"],
                "lint_actions": ["carry_context_only"],
            },
            "style_retry": {
                "strict_retry": True,
                "retry_reasons": ["duplicate_restatement"],
                "bad_output_to_avoid": "모델이 할 수 있는 것 중 하나는 이미지와 텍스트의 연관성만으로 학습할 수 있다는 점이고, 이미지와 텍스트의 연관성만으로요.",
                "note": "When a detector-miss invocation surfaces a continuation tail, keep the protected cue verbatim and rewrite only cue 2 into the missing tail fragment.",
            },
            "source_cues": [
                {
                    "cue_index": 191,
                    "start": "00:07:58,470",
                    "end": "00:08:02,070",
                    "text": "like model is that it can be trained with just associations",
                    "duration_ms": 3600,
                    "gap_after_ms": 0,
                },
                {
                    "cue_index": 192,
                    "start": "00:08:02,070",
                    "end": "00:08:03,650",
                    "text": "of images and text.",
                    "duration_ms": 1580,
                    "gap_after_ms": 0,
                },
            ],
            "left_context": [],
            "right_context": [],
            "glossary_terms": [],
            "previous_emitted_cues": [
                {"cue_index": 191, "text": "모델이 할 수 있는 것 중 하나는 이미지와 텍스트의 연관성만으로 학습할 수 있다는 점입니다."},
                {"cue_index": 192, "text": "이미지와 텍스트의 연관성만으로 학습할 수 있다는 점입니다."},
            ],
            "offending_cue_indices": [192],
            "protected_cue_indices": [191],
            "preferred_actions": ["restore_missing_tail"],
            "offending_spans": [
                {
                    "cue_index": 192,
                    "cue_indices": [191, 192],
                    "span_text": "이미지와 텍스트의 연관성만으로 학습할 수 있다는 점입니다",
                    "left_text": "모델이 할 수 있는 것 중 하나는 이미지와 텍스트의 연관성만으로 학습할 수 있다는 점입니다",
                    "right_text": "이미지와 텍스트의 연관성만으로 학습할 수 있다는 점입니다",
                    "source_cue_text": "of images and text.",
                    "source_tail_type": "continuation_tail",
                    "issue": "duplicate_restatement",
                    "preferred_action": "restore_missing_tail",
                    "trigger_reason": "detector_miss",
                }
            ],
        }
        strict_continuation_detector_miss_assistant = {
            "emitted_cues": [
                {"cue_index": 191, "text": "모델이 할 수 있는 것 중 하나는 이미지와 텍스트의 연관성만으로 학습할 수 있다는 점입니다."},
                {"cue_index": 192, "text": "이미지와 텍스트의 연관성으로요."},
            ],
            "risk_flags": [],
        }
        messages.append({"role": "user", "content": json.dumps(strict_continuation_detector_miss_user, ensure_ascii=False)})
        messages.append({"role": "assistant", "content": json.dumps(strict_continuation_detector_miss_assistant, ensure_ascii=False)})
        strict_continuation_overedit_user = {
            "task": "subtitle_phase1_translate",
            "boundary_lint": {
                "lint_flags": ["dependent_end"],
                "lint_actions": ["carry_context_only"],
            },
            "style_retry": {
                "strict_retry": True,
                "retry_reasons": ["duplicate_restatement"],
                "bad_output_to_avoid": "이게 어떻게 작동하는지 보려면, 신경망을 다르게 생각하는 게 도움이 됩니다, 즉, 신경망을",
                "note": "Do not add a discourse-marker head or collapse the continuation into a too-short fragment. Keep the remaining tail itself.",
            },
            "source_cues": [
                {
                    "cue_index": 1366,
                    "start": "00:01:30,000",
                    "end": "00:01:31,900",
                    "text": "So to see how this works, it's useful to think",
                    "duration_ms": 1900,
                    "gap_after_ms": 40,
                },
                {
                    "cue_index": 1367,
                    "start": "00:01:31,940",
                    "end": "00:01:33,300",
                    "text": "of your neural network in a different way,",
                    "duration_ms": 1360,
                    "gap_after_ms": 0,
                },
            ],
            "left_context": [],
            "right_context": [],
            "glossary_terms": [],
            "previous_emitted_cues": [
                {"cue_index": 1366, "text": "이게 어떻게 작동하는지 보려면, 신경망을 다르게 생각하는 게 도움이 됩니다,"},
                {"cue_index": 1367, "text": "신경망을 다르게 생각하는 게 도움이 됩니다,"},
            ],
            "offending_cue_indices": [1367],
            "protected_cue_indices": [1366],
            "preferred_actions": ["restore_missing_tail"],
            "offending_spans": [
                {
                    "cue_index": 1367,
                    "cue_indices": [1366, 1367],
                    "span_text": "신경망을 다르게 생각하는 게 도움이 됩니다,",
                    "left_text": "신경망을 다르게 생각하는 게 도움이 됩니다,",
                    "right_text": "신경망을 다르게 생각하는 게 도움이 됩니다,",
                    "source_cue_text": "of your neural network in a different way,",
                    "source_tail_type": "continuation_tail",
                    "issue": "duplicate_restatement",
                    "preferred_action": "restore_missing_tail",
                }
            ],
        }
        strict_continuation_overedit_assistant = {
            "emitted_cues": [
                {"cue_index": 1366, "text": "이게 어떻게 작동하는지 보려면, 신경망을 다르게 생각하는 게 도움이 됩니다,"},
                {"cue_index": 1367, "text": "여러분의 신경망을 다른 방식으로,"},
            ],
            "risk_flags": [],
        }
        messages.append({"role": "user", "content": json.dumps(strict_continuation_overedit_user, ensure_ascii=False)})
        messages.append({"role": "assistant", "content": json.dumps(strict_continuation_overedit_assistant, ensure_ascii=False)})
        strict_continuation_empty_tail_user = {
            "task": "subtitle_phase1_translate",
            "boundary_lint": {
                "lint_flags": [],
                "lint_actions": [],
            },
            "style_retry": {
                "strict_retry": True,
                "retry_reasons": ["duplicate_restatement"],
                "bad_output_to_avoid": "시각 장면의 풍부함 덕분에 아주 밀도 높은 장면 그래프를 만들 수 있습니다. 시각 장면의 풍부함 덕분에",
                "note": "Do not delete the continuation tail when the source still contains a recoverable noun phrase. Keep a short non-empty continuation fragment.",
            },
            "source_cues": [
                {
                    "cue_index": 1410,
                    "start": "00:01:34,000",
                    "end": "00:01:35,900",
                    "text": "because of the richness",
                    "duration_ms": 1900,
                    "gap_after_ms": 40,
                },
                {
                    "cue_index": 1411,
                    "start": "00:01:35,940",
                    "end": "00:01:37,100",
                    "text": "of the visual scene.",
                    "duration_ms": 1160,
                    "gap_after_ms": 0,
                },
            ],
            "left_context": [],
            "right_context": [],
            "glossary_terms": [],
            "previous_emitted_cues": [
                {"cue_index": 1410, "text": "시각 장면의 풍부함 덕분에 아주 밀도 높은 장면 그래프를 만들 수 있습니다."},
                {"cue_index": 1411, "text": "시각 장면의 풍부함 덕분에"},
            ],
            "offending_cue_indices": [1411],
            "protected_cue_indices": [1410],
            "preferred_actions": ["restore_missing_tail"],
            "offending_spans": [
                {
                    "cue_index": 1411,
                    "cue_indices": [1410, 1411],
                    "span_text": "시각 장면의 풍부함 덕분에",
                    "left_text": "시각 장면의 풍부함 덕분에 아주 밀도 높은 장면 그래프를 만들 수 있습니다.",
                    "right_text": "시각 장면의 풍부함 덕분에",
                    "source_cue_text": "of the visual scene.",
                    "source_tail_type": "continuation_tail",
                    "issue": "duplicate_restatement",
                    "preferred_action": "restore_missing_tail",
                }
            ],
        }
        strict_continuation_empty_tail_assistant = {
            "emitted_cues": [
                {"cue_index": 1410, "text": "시각 장면의 풍부함 덕분에 아주 밀도 높은 장면 그래프를 만들 수 있습니다."},
                {"cue_index": 1411, "text": "시각 장면 자체의."},
            ],
            "risk_flags": [],
        }
        messages.append({"role": "user", "content": json.dumps(strict_continuation_empty_tail_user, ensure_ascii=False)})
        messages.append({"role": "assistant", "content": json.dumps(strict_continuation_empty_tail_assistant, ensure_ascii=False)})
    if strict_style_retry and prompt_profile == "fragment_preserving_v3":
        strict_continuation_local_meaning_user = {
            "task": "subtitle_phase1_translate",
            "boundary_lint": {
                "lint_flags": [],
                "lint_actions": [],
            },
            "style_retry": {
                "strict_retry": True,
                "retry_reasons": ["duplicate_restatement"],
                "bad_output_to_avoid": "신경계의 발달을 이끌었고, 지능의 발달도 마찬가지입니다. 지능의 발달에 관해서입니다.",
                "note": "If cue 1 already carries the predicate, cue 2 should restore only the leftover continuation idea such as '지능도 마찬가지로요.' instead of closing with another predicate.",
            },
            "source_cues": [
                {
                    "cue_index": 1501,
                    "start": "00:00:10,000",
                    "end": "00:00:11,600",
                    "text": "and drove the development of nervous systems,",
                    "duration_ms": 1600,
                    "gap_after_ms": 40,
                },
                {
                    "cue_index": 1502,
                    "start": "00:00:11,640",
                    "end": "00:00:12,600",
                    "text": "and intelligence as well.",
                    "duration_ms": 960,
                    "gap_after_ms": 0,
                },
            ],
            "left_context": [],
            "right_context": [],
            "glossary_terms": [],
            "previous_emitted_cues": [
                {"cue_index": 1501, "text": "신경계의 발달을 이끌었고,"},
                {"cue_index": 1502, "text": "지능의 발달입니다."},
            ],
            "offending_cue_indices": [1502],
            "protected_cue_indices": [1501],
            "preferred_actions": ["restore_missing_tail"],
            "offending_spans": [
                {
                    "cue_index": 1502,
                    "cue_indices": [1501, 1502],
                    "span_text": "지능의 발달입니다.",
                    "left_text": "신경계의 발달을 이끌었고,",
                    "right_text": "지능의 발달입니다.",
                    "source_cue_text": "and intelligence as well.",
                    "source_tail_type": "continuation_tail",
                    "issue": "duplicate_restatement",
                    "preferred_action": "restore_missing_tail",
                }
            ],
        }
        strict_continuation_local_meaning_assistant = {
            "emitted_cues": [
                {"cue_index": 1501, "text": "신경계의 발달을 이끌었고,"},
                {"cue_index": 1502, "text": "지능도 마찬가지로요."},
            ],
            "risk_flags": [],
        }
        messages.append({"role": "user", "content": json.dumps(strict_continuation_local_meaning_user, ensure_ascii=False)})
        messages.append({"role": "assistant", "content": json.dumps(strict_continuation_local_meaning_assistant, ensure_ascii=False)})

        strict_continuation_from_tail_user = {
            "task": "subtitle_phase1_translate",
            "boundary_lint": {
                "lint_flags": [],
                "lint_actions": [],
            },
            "style_retry": {
                "strict_retry": True,
                "retry_reasons": ["duplicate_restatement"],
                "bad_output_to_avoid": "그래서 이 학습된 모델은 직접 학습합니다. 이 실제 세계의 상호작용에서 직접 학습합니다.",
                "note": "For a from-tail, keep the source-side origin phrase instead of repeating the predicate from cue 1.",
            },
            "source_cues": [
                {
                    "cue_index": 1601,
                    "start": "00:00:20,000",
                    "end": "00:00:21,700",
                    "text": "So this learned model can learn directly",
                    "duration_ms": 1700,
                    "gap_after_ms": 40,
                },
                {
                    "cue_index": 1602,
                    "start": "00:00:21,740",
                    "end": "00:00:22,900",
                    "text": "from these real-world interactions.",
                    "duration_ms": 1160,
                    "gap_after_ms": 0,
                },
            ],
            "left_context": [],
            "right_context": [],
            "glossary_terms": [],
            "previous_emitted_cues": [
                {"cue_index": 1601, "text": "그래서 이 학습된 모델은 직접 학습합니다."},
                {"cue_index": 1602, "text": "이 실제 세계의 상호작용에서 직접 학습합니다."},
            ],
            "offending_cue_indices": [1602],
            "protected_cue_indices": [1601],
            "preferred_actions": ["restore_missing_tail"],
            "offending_spans": [
                {
                    "cue_index": 1602,
                    "cue_indices": [1601, 1602],
                    "span_text": "이 실제 세계의 상호작용에서 직접 학습합니다.",
                    "left_text": "그래서 이 학습된 모델은 직접 학습합니다.",
                    "right_text": "이 실제 세계의 상호작용에서 직접 학습합니다.",
                    "source_cue_text": "from these real-world interactions.",
                    "source_tail_type": "continuation_tail",
                    "issue": "duplicate_restatement",
                    "preferred_action": "restore_missing_tail",
                }
            ],
        }
        strict_continuation_from_tail_assistant = {
            "emitted_cues": [
                {"cue_index": 1601, "text": "그래서 이 학습된 모델은 직접 학습합니다."},
                {"cue_index": 1602, "text": "이런 실제 세계의 상호작용에서부터요."},
            ],
            "risk_flags": [],
        }
        messages.append({"role": "user", "content": json.dumps(strict_continuation_from_tail_user, ensure_ascii=False)})
        messages.append({"role": "assistant", "content": json.dumps(strict_continuation_from_tail_assistant, ensure_ascii=False)})

        strict_continuation_modifier_tail_user = {
            "task": "subtitle_phase1_translate",
            "boundary_lint": {
                "lint_flags": [],
                "lint_actions": [],
            },
            "style_retry": {
                "strict_retry": True,
                "retry_reasons": ["duplicate_restatement"],
                "bad_output_to_avoid": "해야 할 것은 계산 그래프에 들어갈 수 있는 새로운 노드 유형 몇 가지를 추가하는 것입니다. 계산 그래프에 들어갈 수 있는 새로운 노드 유형 몇 가지를 추가하는 것입니다.",
                "note": "When cue 1 already states the main proposition, cue 2 should restore only the modifier tail, not restate the whole noun phrase predicate.",
            },
            "source_cues": [
                {
                    "cue_index": 1701,
                    "start": "00:00:30,000",
                    "end": "00:00:31,900",
                    "text": "and all you need to do is add a few new types of nodes",
                    "duration_ms": 1900,
                    "gap_after_ms": 40,
                },
                {
                    "cue_index": 1702,
                    "start": "00:00:31,940",
                    "end": "00:00:33,000",
                    "text": "that can go into your computation graph.",
                    "duration_ms": 1060,
                    "gap_after_ms": 0,
                },
            ],
            "left_context": [],
            "right_context": [],
            "glossary_terms": [],
            "previous_emitted_cues": [
                {"cue_index": 1701, "text": "해야 할 것은 새로운 노드 유형 몇 가지를 추가하는 것입니다."},
                {"cue_index": 1702, "text": "계산 그래프에 들어갈 수 있는 새로운 노드 유형 몇 가지를 추가하는 것입니다."},
            ],
            "offending_cue_indices": [1702],
            "protected_cue_indices": [1701],
            "preferred_actions": ["restore_missing_tail"],
            "offending_spans": [
                {
                    "cue_index": 1702,
                    "cue_indices": [1701, 1702],
                    "span_text": "계산 그래프에 들어갈 수 있는 새로운 노드 유형 몇 가지를 추가하는 것입니다.",
                    "left_text": "해야 할 것은 새로운 노드 유형 몇 가지를 추가하는 것입니다.",
                    "right_text": "계산 그래프에 들어갈 수 있는 새로운 노드 유형 몇 가지를 추가하는 것입니다.",
                    "source_cue_text": "that can go into your computation graph.",
                    "source_tail_type": "continuation_tail",
                    "issue": "duplicate_restatement",
                    "preferred_action": "restore_missing_tail",
                }
            ],
        }
        strict_continuation_modifier_tail_assistant = {
            "emitted_cues": [
                {"cue_index": 1701, "text": "해야 할 것은 새로운 노드 유형 몇 가지를 추가하는 것입니다."},
                {"cue_index": 1702, "text": "계산 그래프에 들어갈 수 있는 것들이요."},
            ],
            "risk_flags": [],
        }
        messages.append({"role": "user", "content": json.dumps(strict_continuation_modifier_tail_user, ensure_ascii=False)})
        messages.append({"role": "assistant", "content": json.dumps(strict_continuation_modifier_tail_assistant, ensure_ascii=False)})
    return messages


def build_phase1_system_prompt(
    config: RuntimeConfig,
    strict_style_retry: bool = False,
    style_retry_reasons: List[str] | None = None,
    strict_retry_mode: str = "full_block",
    prompt_profile: str | None = None,
) -> str:
    parts = [
        "You are a precise subtitle translator.",
        "Use context only to interpret the current block.",
        "Priority order: cue count/order and anchor preservation > fragment shape preservation > natural Korean > lecture tone.",
        "Return exactly one emitted cue for each input cue.",
        "Keep cue order and cue count unchanged.",
        "Do not add or remove meaning.",
        "Do not leave the first cue empty unless the source cue is empty.",
        "Preserve numbers, formulas, brackets, and glossary terms.",
        "Produce natural Korean only after preserving fragment shape and timing anchors.",
        "If there is a risk of poor alignment or readability, report short risk_flags.",
        "The input may include boundary lint flags and lint actions.",
        "Treat dependent_start as a diagnostic label only; do not change behavior solely because it is present.",
        "If lint_flags include comparison_midstart, treat the block as a mid-thought fragment and do not invent a missing lead-in.",
        "If lint_flags include dependent_end, do not over-close the thought with added content that is not present.",
        "If lint_flags include qa_fragment, keep acknowledgements short and do not expand or duplicate them.",
        "If lint_flags include numeric_orphan, preserve numbers, ratios, symbols, and abbreviations exactly and do not add surrounding explanation.",
        "If lint_actions include carry_context_only, keep the current cue boundary and translate as a context-dependent fragment instead of forcing full local completeness.",
        "For fragmentary blocks, do not add unsupported discourse-marker heads such as '즉,' or '다시 말해,' unless the source directly supports them.",
        "For fragmentary blocks, do not add generic explanatory closings such as '...라는 뜻입니다', '...거죠', '...겁니다', or '...라고 볼 수 있습니다' unless the source directly supports them.",
        "Do not restate the same meaning twice in a second paraphrastic sentence or clause.",
        f"Allowed risk_flags are: {', '.join(RISK_FLAG_ENUM)}.",
    ]
    if strict_style_retry:
        retry_reasons = ", ".join(style_retry_reasons or ["style_warning"])
        parts.extend(
            [
                f"This is a strict style retry after style warnings: {retry_reasons}.",
                "On this retry, keep the Korean unfinished whenever the source thought is unfinished.",
                "It is better to end with a comma, dash, or bare fragment than to add a lecture-style sentence ending that is not supported.",
                "Do not add a second sentence or clause that only rephrases the first.",
                "Rewrite only the offending cues or offending spans when possible.",
                "Keep protected cues unchanged unless preserving them would directly prevent fixing the listed style problem.",
                "If preferred_actions include drop_head_marker, delete the unsupported discourse marker at the cue head and keep the remaining fragment.",
                "If preferred_actions include trim_explanatory_tail, delete unsupported trailing explanation instead of replacing it with another explanatory tail.",
                "If preferred_actions include delete_repeat_local, remove the repeated local proposition instead of paraphrasing it again.",
                "If preferred_actions include restore_missing_tail, rewrite the offending cue so it restores that cue's missing local source meaning instead of repeating the previous cue.",
                "When offending_spans include source_tail_type, preserve that tail shape instead of normalizing it into a generic sentence.",
                "When offending_spans include trigger_reason=detector_miss, keep the protected cue verbatim whenever possible and fix only the offending cue.",
                "For purpose_tail, restore a purpose fragment such as '...하기 위해', '...하려고', or '...하도록' and avoid a copular ending like '...입니다'.",
                "For that_clause_tail, keep a subordinate-clause shape such as '...라고', '...라는', '...다고', '...하면', or '...때문에' rather than a standalone closed sentence.",
                "For relative_clause_tail, keep a modifier shape such as '...하는', '...되는', '...인', or '...할' rather than a standalone sentence.",
                "For comparison_tail, keep a comparison shape such as '...처럼', '...같이', '...보다', '...만큼', or '...에 비해' instead of closing the thought.",
                "For continuation_tail, prefer a connective continuation over a full sentence ending when the source tail is only a connective phrase.",
                "For continuation_tail, if the protected cue already carries the main proposition, keep only the leftover local tail in the offending cue instead of restating the whole proposition.",
                "For continuation_tail, if the protected cue already ends with an adverbial anchor such as '바로', do not repeat the same anchor at the start of the offending cue.",
                "For continuation_tail, avoid generic closed endings such as '...원합니다', '...입니다', or '...말입니다' when a shorter connective tail would preserve the source shape better.",
                "For continuation_tail, never start the offending cue with unsupported discourse-marker heads like '즉,' or '다시 말해,'.",
                "For continuation_tail, do not collapse the offending cue to an empty string or a bare one-word fragment when the source tail still contains recoverable local meaning.",
                "For continuation_tail, preserve at least one concrete lexical noun or modifier from the source tail in the offending cue instead of deleting it entirely.",
                "For continuation_tail short tails such as 'of X', 'with X', or 'from X', prefer a short noun-phrase continuation like 'X의', 'X 쪽의', or 'X 자체의' over an empty cue.",
                "Keep unaffected cues as close as possible to the previous anchor-preserving wording.",
            ]
        )
        if prompt_profile == "fragment_preserving_v3":
            parts.extend(
                [
                    "For continuation_tail local restoration, if cue 1 already carries the main proposition, cue 2 should usually restore only the leftover local tail phrase or modifier.",
                    "Do not turn that leftover tail into a fresh predicate like '...입니다', '...합니다', or '...라는 점입니다' unless the source tail itself is a full predicate.",
                    "If a continuation tail starts with 'from', 'of', or 'that can', keep that source-side relation visible in cue 2 instead of repeating the full proposition from cue 1.",
                ]
            )
        if strict_retry_mode == "offending_cue_only":
            parts.extend(
                [
                    "When style_retry.retry_contract is offending_cue_only, return only offending_cue_rewrites.",
                    "Do not return rewritten protected cues.",
                    "Rewrite only the listed offending cue indices and leave every protected cue implicit and unchanged.",
                ]
            )
    if config.translation_context:
        parts.append(f"Context: {compact_spaces(config.translation_context)}")
    if config.translation_style:
        parts.append(f"Style guidance: {compact_spaces(config.translation_style)}")
    return " ".join(parts)


def build_repair_system_prompt(config: RuntimeConfig) -> str:
    return build_repair_system_prompt_for_profile(config, "baseline")


def build_repair_system_prompt_for_profile(config: RuntimeConfig, repair_profile: str) -> str:
    parts = [
        "You are a subtitle repair editor performing bounded rewrites only.",
        "Rewrite only enough to improve alignment and readability.",
        "Keep cue count and cue order unchanged.",
        "Do not add or remove meaning.",
        "Preserve numbers, formulas, brackets, and glossary terms.",
        "Prefer not to leave the first cue empty.",
        "Improve natural Korean and line readability while respecting timing anchors.",
        f"Hard line constraints: each cue must fit within at most {config.max_lines_per_cue} line(s), "
        f"with at most {config.max_chars_per_line} characters per line after runtime wrapping.",
        "When failure_reasons include line_overflow, compact the cue text enough to satisfy those hard line constraints; do not rely on manual line breaks alone.",
        "For overlong single-cue subtitles, remove filler and repeated lecture phrasing while preserving source meaning and technical anchors.",
        "Fix only the listed failures.",
        "Leave unaffected cues unchanged whenever possible.",
        "Change as little as possible.",
        "Boundary lint flags may indicate that the source block is fragmentary; do not invent missing context while repairing readability.",
        f"Allowed risk_flags are: {', '.join(RISK_FLAG_ENUM)}.",
        "Return only repaired emitted cues and optional risk flags.",
    ]
    if repair_profile == "compact_technical_fragment_v1":
        parts.extend(
            [
                "For single-cue technical fragments with line overflow, compress repeated predicates into a compact list-like fragment instead of repeating full sentence endings.",
                "Keep technical model names, numbers, and units exactly.",
                "Prefer commas, parallel phrasing, or an unfinished fragment ending over repeating '...필요합니다' for every item.",
                "If the source cue itself is unfinished, keep the repaired cue unfinished rather than forcing a closed sentence.",
                "The goal is to fit the same meaning into readable subtitle lines with minimal wording.",
            ]
        )
    if config.translation_context:
        parts.append(f"Context: {compact_spaces(config.translation_context)}")
    if config.translation_style:
        parts.append(f"Style guidance: {compact_spaces(config.translation_style)}")
    return " ".join(parts)


def _repair_profile_for_request(config: RuntimeConfig, request: RepairRequest) -> str:
    if config.repair_policy != "compact_technical_fragment_v1":
        return "baseline"
    if len(request.block.cues) != 1:
        return "baseline"
    if "line_overflow" not in request.failure_reasons:
        return "baseline"
    if "dependent_end" not in request.block.lint_reasons:
        return "baseline"
    source_text = normalize_text(request.block.cues[0].text).casefold()
    phase1_text = " ".join(normalize_text(cue.text).casefold() for cue in request.phase1_result.emitted_cues)
    technical_hits = {
        token.casefold()
        for token in _REPAIR_TECHNICAL_HINT_RE.findall(f"{source_text} {phase1_text}")
        if (
            token.casefold() in _REPAIR_TECHNICAL_HINTS
            or ("-" in token and any(ch.isdigit() for ch in token))
            or token.casefold().endswith("flops")
        )
    }
    if len(technical_hits) < 3:
        return "baseline"
    return "compact_technical_fragment_v1"


def _repair_example_messages(repair_profile: str) -> List[dict]:
    if repair_profile != "compact_technical_fragment_v1":
        return []
    user_payload = {
        "task": "subtitle_phase2_repair",
        "boundary_lint": {
            "lint_flags": ["dependent_end"],
            "lint_actions": [],
        },
        "source_cues": [
            {
                "cue_index": 405,
                "start": "00:00:10,000",
                "end": "00:00:18,640",
                "text": "So for AlexNet, it takes 0.7 GFLOPS. For VGG-16, it takes like 13.6 GFLOPS. But for C3D,",
                "duration_ms": 8640,
                "gap_after_ms": 0,
            }
        ],
        "phase1_emitted_cues": [
            {
                "cue_index": 405,
                "text": "그래서 AlexNet은 0.7 GFLOPS가 필요합니다. VGG-16은 약 13.6 GFLOPS가 필요하고, C3D는",
            }
        ],
        "failure_reasons": ["line_overflow"],
        "line_constraints": {
            "max_lines_per_cue": 2,
            "max_chars_per_line": 24,
        },
        "glossary_terms": [],
    }
    assistant_payload = {
        "emitted_cues": [
            {
                "cue_index": 405,
                "text": "AlexNet은 0.7 GFLOPS,\nVGG-16은 13.6 GFLOPS, C3D는",
            }
        ],
        "risk_flags": [],
    }
    return [
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        {"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=False)},
    ]


class BaseTranslator(ABC):
    @abstractmethod
    def translate_block(self, request: TranslationRequest) -> PhaseTranslationResult:
        raise NotImplementedError

    @abstractmethod
    def repair_block(self, request: RepairRequest) -> PhaseTranslationResult:
        raise NotImplementedError


class OpenAIChatTranslator(BaseTranslator):
    def __init__(
        self,
        config: RuntimeConfig,
        phase1_model: str | None = None,
        repair_model: str | None = None,
        base_url: str = "https://api.openai.com/v1",
    ):
        self.config = config
        self.phase1_model = phase1_model or config.phase1_model
        self.repair_model = repair_model or config.repair_model
        self.base_url = base_url.rstrip("/")
        self.api_key = config.openai_api_key
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to the environment or .env file.")
        self.phase1_system_prompt = build_phase1_system_prompt(config)
        self.repair_system_prompt = build_repair_system_prompt_for_profile(config, "baseline")
        self.phase1_example_messages = _phase1_example_messages(config.phase1_prompt_profile)
        self.phase1_strict_example_messages = _phase1_example_messages(
            config.phase1_prompt_profile,
            strict_style_retry=True,
            strict_retry_mode="full_block",
        )
        self.phase1_strict_offending_only_example_messages = _phase1_example_messages(
            config.phase1_prompt_profile,
            strict_style_retry=True,
            strict_retry_mode="offending_cue_only",
        )
        self.repair_example_messages = _repair_example_messages("baseline")
        self.session = requests.Session()

    def _effective_prompt_profile(self, request: TranslationRequest) -> str:
        return request.prompt_profile_override or self.config.phase1_prompt_profile

    def _phase1_example_messages_for_request(self, request: TranslationRequest) -> List[dict]:
        prompt_profile = self._effective_prompt_profile(request)
        strict_style_retry = request.strict_style_retry
        strict_retry_mode = request.strict_retry_mode
        if prompt_profile == self.config.phase1_prompt_profile:
            if strict_style_retry and strict_retry_mode == "offending_cue_only":
                return self.phase1_strict_offending_only_example_messages
            if strict_style_retry:
                return self.phase1_strict_example_messages
            return self.phase1_example_messages
        return _phase1_example_messages(
            prompt_profile,
            strict_style_retry=strict_style_retry,
            strict_retry_mode=strict_retry_mode,
        )

    def _effective_repair_profile(self, request: RepairRequest) -> str:
        return _repair_profile_for_request(self.config, request)

    def _repair_example_messages_for_request(self, request: RepairRequest) -> List[dict]:
        repair_profile = self._effective_repair_profile(request)
        if repair_profile == "baseline":
            return self.repair_example_messages
        return _repair_example_messages(repair_profile)

    def _payload_for_phase1(self, request: TranslationRequest) -> dict:
        block = request.block
        cues = block.cues
        payload = {
            "task": "subtitle_phase1_translate",
            "boundary_lint": {
                "lint_flags": block.lint_reasons,
                "lint_actions": block.lint_actions,
            },
            "source_cues": [
                {
                    "cue_index": cue.index,
                    "start": cue.start,
                    "end": cue.end,
                    "text": cue.text,
                    "duration_ms": max(ts_to_ms(cue.end) - ts_to_ms(cue.start), 1),
                    "gap_after_ms": max(ts_to_ms(cues[idx + 1].start) - ts_to_ms(cue.end), 0) if idx + 1 < len(cues) else 0,
                }
                for idx, cue in enumerate(cues)
            ],
            "left_context": block.previous_source_sentences,
            "right_context": block.next_source_sentences,
            "glossary_terms": [
                {"source": term.source, "target": term.target, "note": term.note, "mode": term.mode}
                for term in request.glossary_terms
            ],
        }
        if request.strict_style_retry:
            payload["style_retry"] = {
                "strict_retry": True,
                "retry_contract": request.strict_retry_mode,
                "retry_reasons": request.style_retry_reasons,
                "previous_emitted_cues": [
                    {"cue_index": cue.cue_index, "text": cue.text}
                    for cue in request.previous_emitted_cues
                ],
                "protected_cue_indices": request.protected_cue_indices,
                "offending_cue_indices": request.offending_cue_indices,
                "offending_spans": request.offending_spans,
                "preferred_actions": request.preferred_actions,
            }
        return payload

    def _payload_for_repair(self, request: RepairRequest) -> dict:
        block = request.block
        cues = block.cues
        return {
            "task": "subtitle_phase2_repair",
            "boundary_lint": {
                "lint_flags": block.lint_reasons,
                "lint_actions": block.lint_actions,
            },
            "source_cues": [
                {
                    "cue_index": cue.index,
                    "start": cue.start,
                    "end": cue.end,
                    "text": cue.text,
                    "duration_ms": max(ts_to_ms(cue.end) - ts_to_ms(cue.start), 1),
                    "gap_after_ms": max(ts_to_ms(cues[idx + 1].start) - ts_to_ms(cue.end), 0) if idx + 1 < len(cues) else 0,
                }
                for idx, cue in enumerate(cues)
            ],
            "phase1_emitted_cues": [
                {"cue_index": cue.cue_index, "text": cue.text}
                for cue in request.phase1_result.emitted_cues
            ],
            "failure_reasons": request.failure_reasons,
            "line_constraints": {
                "max_lines_per_cue": self.config.max_lines_per_cue,
                "max_chars_per_line": self.config.max_chars_per_line,
            },
            "glossary_terms": [
                {"source": term.source, "target": term.target, "note": term.note, "mode": term.mode}
                for term in request.glossary_terms
            ],
        }

    def build_chat_completion_request_body(
        self,
        model: str,
        system_prompt: str,
        payload: dict,
        response_format: dict,
        temperature: float,
        example_messages: List[dict] | None = None,
    ) -> dict:
        return {
            "model": model,
            "messages": (
                [{"role": "system", "content": system_prompt}]
                + list(example_messages or [])
                + [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
            ),
            "temperature": temperature,
            "response_format": response_format,
        }

    def parse_chat_completion_response_body(self, payload: dict) -> PhaseTranslationResult:
        message = payload["choices"][0]["message"]
        if message.get("refusal"):
            raise RuntimeError(f"Model refused request: {message['refusal']}")
        content = _extract_message_text(message.get("content"))
        parsed = json.loads(content)
        emitted_cues = [
            EmittedCue(
                cue_index=int(item["cue_index"]),
                text=normalize_text(str(item["text"])),
            )
            for item in parsed.get("emitted_cues", [])
        ]
        risk_flags = [normalize_text(str(flag)) for flag in parsed.get("risk_flags", []) if normalize_text(str(flag))]
        return PhaseTranslationResult(emitted_cues=emitted_cues, risk_flags=list(dict.fromkeys(risk_flags)))

    def parse_offending_only_phase_response_body(
        self,
        payload: dict,
        request: TranslationRequest,
    ) -> PhaseTranslationResult:
        message = payload["choices"][0]["message"]
        if message.get("refusal"):
            raise RuntimeError(f"Model refused request: {message['refusal']}")
        content = _extract_message_text(message.get("content"))
        parsed = json.loads(content)
        rewrites = parsed.get("offending_cue_rewrites", [])
        risk_flags = [normalize_text(str(flag)) for flag in parsed.get("risk_flags", []) if normalize_text(str(flag))]
        return _splice_offending_cue_rewrites(request, rewrites, risk_flags)

    def build_phase1_request_body(self, request: TranslationRequest) -> dict:
        strict_style_retry = request.strict_style_retry
        prompt_profile = self._effective_prompt_profile(request)
        return self.build_chat_completion_request_body(
            model=self.phase1_model,
            system_prompt=(
                build_phase1_system_prompt(
                    self.config,
                    strict_style_retry=True,
                    style_retry_reasons=request.style_retry_reasons,
                    strict_retry_mode=request.strict_retry_mode,
                    prompt_profile=prompt_profile,
                )
                if strict_style_retry
                else (
                    self.phase1_system_prompt
                    if prompt_profile == self.config.phase1_prompt_profile
                    else build_phase1_system_prompt(self.config, prompt_profile=prompt_profile)
                )
            ),
            payload=self._payload_for_phase1(request),
            response_format=(
                _build_offending_retry_schema(
                    cue_count=len(request.offending_cue_indices),
                    schema_name="subtitle_phase1_strict_offending_only",
                )
                if strict_style_retry and request.strict_retry_mode == "offending_cue_only"
                else _build_phase_schema(
                    cue_count=len(request.block.cues),
                    schema_name="subtitle_phase1_strict" if strict_style_retry else "subtitle_phase1",
                )
            ),
            temperature=0.0 if strict_style_retry else self.config.phase1_temperature,
            example_messages=self._phase1_example_messages_for_request(request),
        )

    def build_repair_request_body(self, request: RepairRequest) -> dict:
        repair_profile = self._effective_repair_profile(request)
        return self.build_chat_completion_request_body(
            model=self.repair_model,
            system_prompt=(
                self.repair_system_prompt
                if repair_profile == "baseline"
                else build_repair_system_prompt_for_profile(self.config, repair_profile)
            ),
            payload=self._payload_for_repair(request),
            response_format=_build_phase_schema(
                cue_count=len(request.block.cues),
                schema_name="subtitle_phase2_repair",
            ),
            temperature=self.config.repair_temperature,
            example_messages=self._repair_example_messages_for_request(request),
        )

    def _structured_completion_with_parser(
        self,
        model: str,
        system_prompt: str,
        payload: dict,
        response_format: dict,
        temperature: float,
        parser,
        example_messages: List[dict] | None = None,
    ) -> PhaseTranslationResult:
        data = self.build_chat_completion_request_body(
            model=model,
            system_prompt=system_prompt,
            payload=payload,
            response_format=response_format,
            temperature=temperature,
            example_messages=example_messages,
        )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_exc: Exception | None = None
        for attempt in range(self.config.request_max_attempts):
            try:
                response = self.session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=self.config.request_timeout,
                )
                if response.status_code in {429, 500, 502, 503, 504}:
                    retry_after = response.headers.get("retry-after")
                    base_delay = min(
                        self.config.request_backoff_max_seconds,
                        self.config.request_backoff_min_seconds * (2 ** attempt),
                    )
                    jitter_delay = random.uniform(
                        self.config.request_backoff_min_seconds,
                        max(self.config.request_backoff_min_seconds, base_delay),
                    )
                    delay = float(retry_after) if retry_after else jitter_delay
                    if attempt + 1 >= self.config.request_max_attempts:
                        response.raise_for_status()
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                return parser(response.json())
            except requests.HTTPError as exc:
                last_exc = exc
                raise
            except requests.RequestException as exc:
                last_exc = exc
                if attempt + 1 >= self.config.request_max_attempts:
                    raise
                base_delay = min(
                    self.config.request_backoff_max_seconds,
                    self.config.request_backoff_min_seconds * (2 ** attempt),
                )
                time.sleep(
                    random.uniform(
                        self.config.request_backoff_min_seconds,
                        max(self.config.request_backoff_min_seconds, base_delay),
                    )
                )
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Structured completion failed without a concrete exception.")

    def _structured_completion(
        self,
        model: str,
        system_prompt: str,
        payload: dict,
        response_format: dict,
        temperature: float,
        example_messages: List[dict] | None = None,
    ) -> PhaseTranslationResult:
        return self._structured_completion_with_parser(
            model=model,
            system_prompt=system_prompt,
            payload=payload,
            response_format=response_format,
            temperature=temperature,
            parser=self.parse_chat_completion_response_body,
            example_messages=example_messages,
        )

    def translate_block(self, request: TranslationRequest) -> PhaseTranslationResult:
        strict_style_retry = request.strict_style_retry
        strict_retry_mode = request.strict_retry_mode
        prompt_profile = self._effective_prompt_profile(request)
        system_prompt = (
            build_phase1_system_prompt(
                self.config,
                strict_style_retry=True,
                style_retry_reasons=request.style_retry_reasons,
                strict_retry_mode=strict_retry_mode,
                prompt_profile=prompt_profile,
            )
            if strict_style_retry
            else (
                self.phase1_system_prompt
                if prompt_profile == self.config.phase1_prompt_profile
                else build_phase1_system_prompt(self.config, prompt_profile=prompt_profile)
            )
        )
        response_format = (
            _build_offending_retry_schema(
                cue_count=len(request.offending_cue_indices),
                schema_name="subtitle_phase1_strict_offending_only",
            )
            if strict_style_retry and strict_retry_mode == "offending_cue_only"
            else _build_phase_schema(
                cue_count=len(request.block.cues),
                schema_name="subtitle_phase1_strict" if strict_style_retry else "subtitle_phase1",
            )
        )
        example_messages = self._phase1_example_messages_for_request(request)
        parser = (
            (lambda payload: self.parse_offending_only_phase_response_body(payload, request))
            if strict_style_retry and strict_retry_mode == "offending_cue_only"
            else self.parse_chat_completion_response_body
        )
        temperature = 0.0 if strict_style_retry else self.config.phase1_temperature
        return self._structured_completion_with_parser(
            model=self.phase1_model,
            system_prompt=system_prompt,
            payload=self._payload_for_phase1(request),
            response_format=response_format,
            temperature=temperature,
            parser=parser,
            example_messages=example_messages,
        )

    def repair_block(self, request: RepairRequest) -> PhaseTranslationResult:
        repair_profile = self._effective_repair_profile(request)
        return self._structured_completion(
            model=self.repair_model,
            system_prompt=(
                self.repair_system_prompt
                if repair_profile == "baseline"
                else build_repair_system_prompt_for_profile(self.config, repair_profile)
            ),
            payload=self._payload_for_repair(request),
            response_format=_build_phase_schema(
                cue_count=len(request.block.cues),
                schema_name="subtitle_phase2_repair",
            ),
            temperature=self.config.repair_temperature,
            example_messages=self._repair_example_messages_for_request(request),
        )


def build_translator(
    config: RuntimeConfig,
    model: str | None = None,
    repair_model: str | None = None,
    base_url: str = "https://api.openai.com/v1",
) -> OpenAIChatTranslator:
    return OpenAIChatTranslator(
        config=config,
        phase1_model=model or config.phase1_model,
        repair_model=repair_model or config.repair_model,
        base_url=base_url,
    )
