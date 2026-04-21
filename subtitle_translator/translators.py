from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, List

import requests

from .config import RuntimeConfig
from .models import EmittedCue, PhaseTranslationResult, RepairRequest, TranslationRequest
from .splitting import ts_to_ms
from .text import compact_spaces, normalize_text

RISK_FLAG_ENUM = [
    "fragment_overclosure_risk",
    "duplicate_restatement_risk",
    "english_residual_risk",
    "line_readability_risk",
    "glossary_uncertain",
    "context_uncertain",
]


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
                    {"cue_index": 201, "text": "13% 또는 0.13입니다."}
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


def _phase1_example_messages(prompt_profile: str, strict_style_retry: bool = False) -> List[dict]:
    if prompt_profile == "fragment_preserving_v1":
        examples = _phase1_examples_v1()
    elif prompt_profile == "fragment_preserving_v2":
        examples = _phase1_examples_v2()
    else:
        return []
    messages: List[dict] = []
    for user_payload, assistant_payload in examples:
        messages.append({"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)})
        messages.append({"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=False)})
    if strict_style_retry and prompt_profile == "fragment_preserving_v2":
        strict_user = {
            "task": "subtitle_phase1_translate",
            "boundary_lint": {
                "lint_flags": ["dependent_end"],
                "lint_actions": ["carry_context_only"],
            },
            "style_retry": {
                "strict_retry": True,
                "retry_reasons": ["fragment_overclosure", "duplicate_restatement"],
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
    return messages


def build_phase1_system_prompt(
    config: RuntimeConfig,
    strict_style_retry: bool = False,
    style_retry_reasons: List[str] | None = None,
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
                "If preferred_actions include keep_fragment_open, delete unsupported trailing explanation instead of replacing it with another explanatory tail.",
                "If preferred_actions include delete_repeat, remove the repeated proposition instead of paraphrasing it again.",
                "Keep unaffected cues as close as possible to the previous anchor-preserving wording.",
            ]
        )
    if config.translation_context:
        parts.append(f"Context: {compact_spaces(config.translation_context)}")
    if config.translation_style:
        parts.append(f"Style guidance: {compact_spaces(config.translation_style)}")
    return " ".join(parts)


def build_repair_system_prompt(config: RuntimeConfig) -> str:
    parts = [
        "You are a subtitle repair editor performing bounded rewrites only.",
        "Rewrite only enough to improve alignment and readability.",
        "Keep cue count and cue order unchanged.",
        "Do not add or remove meaning.",
        "Preserve numbers, formulas, brackets, and glossary terms.",
        "Prefer not to leave the first cue empty.",
        "Improve natural Korean and line readability while respecting timing anchors.",
        "Fix only the listed failures.",
        "Leave unaffected cues unchanged whenever possible.",
        "Change as little as possible.",
        "Boundary lint flags may indicate that the source block is fragmentary; do not invent missing context while repairing readability.",
        f"Allowed risk_flags are: {', '.join(RISK_FLAG_ENUM)}.",
        "Return only repaired emitted cues and optional risk flags.",
    ]
    if config.translation_context:
        parts.append(f"Context: {compact_spaces(config.translation_context)}")
    if config.translation_style:
        parts.append(f"Style guidance: {compact_spaces(config.translation_style)}")
    return " ".join(parts)


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
        self.repair_system_prompt = build_repair_system_prompt(config)
        self.phase1_example_messages = _phase1_example_messages(config.phase1_prompt_profile)
        self.phase1_strict_example_messages = _phase1_example_messages(config.phase1_prompt_profile, strict_style_retry=True)
        self.session = requests.Session()

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
                "retry_reasons": request.style_retry_reasons,
                "previous_emitted_cues": [
                    {"cue_index": cue.cue_index, "text": cue.text}
                    for cue in request.previous_emitted_cues
                ],
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
            "glossary_terms": [
                {"source": term.source, "target": term.target, "note": term.note, "mode": term.mode}
                for term in request.glossary_terms
            ],
        }

    def _structured_completion(
        self,
        model: str,
        system_prompt: str,
        payload: dict,
        cue_count: int,
        schema_name: str,
        temperature: float,
        example_messages: List[dict] | None = None,
    ) -> PhaseTranslationResult:
        data = {
            "model": model,
            "messages": (
                [{"role": "system", "content": system_prompt}]
                + list(example_messages or [])
                + [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
            ),
            "temperature": temperature,
            "response_format": _build_phase_schema(cue_count=cue_count, schema_name=schema_name),
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = self.session.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=self.config.request_timeout,
        )
        response.raise_for_status()
        payload = response.json()
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

    def translate_block(self, request: TranslationRequest) -> PhaseTranslationResult:
        strict_style_retry = request.strict_style_retry
        return self._structured_completion(
            model=self.phase1_model,
            system_prompt=(
                build_phase1_system_prompt(
                    self.config,
                    strict_style_retry=True,
                    style_retry_reasons=request.style_retry_reasons,
                )
                if strict_style_retry
                else self.phase1_system_prompt
            ),
            payload=self._payload_for_phase1(request),
            cue_count=len(request.block.cues),
            schema_name="subtitle_phase1_strict" if strict_style_retry else "subtitle_phase1",
            temperature=0.0 if strict_style_retry else self.config.phase1_temperature,
            example_messages=self.phase1_strict_example_messages if strict_style_retry else self.phase1_example_messages,
        )

    def repair_block(self, request: RepairRequest) -> PhaseTranslationResult:
        return self._structured_completion(
            model=self.repair_model,
            system_prompt=self.repair_system_prompt,
            payload=self._payload_for_repair(request),
            cue_count=len(request.block.cues),
            schema_name="subtitle_phase2_repair",
            temperature=self.config.repair_temperature,
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
