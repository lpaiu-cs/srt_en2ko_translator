from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, List

import requests

from .config import RuntimeConfig
from .models import EmittedCue, PhaseTranslationResult, RepairRequest, TranslationRequest
from .text import compact_spaces, normalize_text


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
                        "items": {"type": "string"},
                    },
                },
                "required": ["emitted_cues", "risk_flags"],
            },
        },
    }


def build_phase1_system_prompt(config: RuntimeConfig) -> str:
    parts = [
        "You are a precise subtitle translator.",
        "Use context only to interpret the current block.",
        "Return exactly one emitted cue for each input cue.",
        "Keep cue order and cue count unchanged.",
        "Do not add or remove meaning.",
        "Do not leave the first cue empty unless the source cue is empty.",
        "Preserve numbers, formulas, brackets, and glossary terms.",
        "Produce natural Korean while staying aligned to subtitle timing anchors.",
        "If there is a risk of poor alignment or readability, report short risk_flags.",
    ]
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
        self.session = requests.Session()

    def _payload_for_phase1(self, request: TranslationRequest) -> dict:
        block = request.block
        return {
            "task": "subtitle_phase1_translate",
            "source_cues": [
                {
                    "cue_index": cue.index,
                    "start": cue.start,
                    "end": cue.end,
                    "text": cue.text,
                }
                for cue in block.cues
            ],
            "previous_source_sentences": block.previous_source_sentences,
            "next_source_sentences": block.next_source_sentences,
            "glossary_terms": [
                {"source": term.source, "target": term.target, "note": term.note}
                for term in request.glossary_terms
            ],
        }

    def _payload_for_repair(self, request: RepairRequest) -> dict:
        block = request.block
        return {
            "task": "subtitle_phase2_repair",
            "source_cues": [
                {
                    "cue_index": cue.index,
                    "start": cue.start,
                    "end": cue.end,
                    "text": cue.text,
                }
                for cue in block.cues
            ],
            "phase1_emitted_cues": [
                {"cue_index": cue.cue_index, "text": cue.text}
                for cue in request.phase1_result.emitted_cues
            ],
            "failure_reasons": request.failure_reasons,
            "glossary_terms": [
                {"source": term.source, "target": term.target, "note": term.note}
                for term in request.glossary_terms
            ],
        }

    def _structured_completion(self, model: str, system_prompt: str, payload: dict, cue_count: int, schema_name: str) -> PhaseTranslationResult:
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "temperature": 0.1,
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
        return self._structured_completion(
            model=self.phase1_model,
            system_prompt=self.phase1_system_prompt,
            payload=self._payload_for_phase1(request),
            cue_count=len(request.block.cues),
            schema_name="subtitle_phase1",
        )

    def repair_block(self, request: RepairRequest) -> PhaseTranslationResult:
        return self._structured_completion(
            model=self.repair_model,
            system_prompt=self.repair_system_prompt,
            payload=self._payload_for_repair(request),
            cue_count=len(request.block.cues),
            schema_name="subtitle_phase2_repair",
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
