from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import requests

from .config import RuntimeConfig
from .models import BatchTranslationResult, GlossaryEntry, TranslationRequest
from .text import compact_spaces, normalize_text


def build_translation_system_prompt(config: RuntimeConfig) -> str:
    parts = [
        "You are a precise subtitle translator.",
        "Translate English to Korean succinctly while preserving meaning and tone.",
        "Do NOT add, remove, merge, split, or reorder items.",
        "Return a JSON object with keys 'translations' and 'glossary_updates'.",
        "'translations' must contain exactly one Korean string per input item.",
        "'glossary_updates' must be an array of stable technical terms, names, acronyms, or recurring phrases worth reusing later.",
        "If there are no glossary updates, return an empty array.",
        "If glossary terms are supplied in the user payload, follow them when relevant.",
        "Use previous_translation only for continuity. Do not repeat it verbatim unless the new sentence needs it.",
        "Keep technical terms, names, and acronyms in English when that preserves meaning better.",
    ]
    context = compact_spaces(config.translation_context)
    style = compact_spaces(config.translation_style)
    if context:
        parts.append(f"Context: {context}")
    if style:
        parts.append(f"Style guidance: {style}")
    return " ".join(parts)


def _extract_first_json_value(text: str) -> Any:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            return value
        except ValueError:
            continue
    raise ValueError("No JSON value found in model response")


def _strip_code_fence(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.DOTALL)


def _coerce_glossary_updates(payload: Any) -> List[GlossaryEntry]:
    updates: List[GlossaryEntry] = []
    if not isinstance(payload, list):
        return updates
    for item in payload:
        if not isinstance(item, dict):
            continue
        source = normalize_text(str(item.get("source", "")))
        target = normalize_text(str(item.get("target", "")))
        note = normalize_text(str(item.get("note", "")))
        if source and target:
            updates.append(GlossaryEntry(source=source, target=target, note=note))
    return updates


def parse_translation_response(content: str) -> BatchTranslationResult:
    cleaned = _strip_code_fence(content)
    parsed = _extract_first_json_value(cleaned)
    if isinstance(parsed, dict):
        translations = parsed.get("translations")
        if not isinstance(translations, list):
            raise ValueError("JSON object response did not include a translations list")
        return BatchTranslationResult(
            translations=[normalize_text(str(item)) for item in translations],
            glossary_updates=_coerce_glossary_updates(parsed.get("glossary_updates", [])),
        )
    if isinstance(parsed, list):
        return BatchTranslationResult(
            translations=[normalize_text(str(item)) for item in parsed],
            glossary_updates=[],
        )
    raise ValueError("Unsupported translation response type")


class BaseTranslator(ABC):
    @abstractmethod
    def translate_batch(self, request: TranslationRequest) -> BatchTranslationResult:
        raise NotImplementedError


class OpenAIChatTranslator(BaseTranslator):
    def __init__(self, config: RuntimeConfig, model: str = "gpt-4.1-mini", base_url: str = "https://api.openai.com/v1"):
        self.config = config
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = config.openai_api_key
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to the environment or .env file.")
        self.system_prompt = build_translation_system_prompt(config)
        self.session = requests.Session()

    def _payload(self, request: TranslationRequest) -> dict:
        glossary_terms = [
            {"source": entry.source, "target": entry.target, "note": entry.note}
            for entry in request.glossary_terms
        ]
        return {
            "task": "batch-translate",
            "language_pair": "en→ko",
            "previous_translation": request.previous_translation,
            "glossary_terms": glossary_terms,
            "items": request.sentences,
        }

    def translate_batch(self, request: TranslationRequest) -> BatchTranslationResult:
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(self._payload(request), ensure_ascii=False)},
            ],
            "temperature": 0.2,
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
        content = payload["choices"][0]["message"]["content"].strip()
        try:
            return parse_translation_response(content)
        except Exception:
            lines = [normalize_text(line) for line in _strip_code_fence(content).splitlines() if line.strip()]
            return BatchTranslationResult(translations=lines, glossary_updates=[])


def build_translator(config: RuntimeConfig, model: str, base_url: str) -> OpenAIChatTranslator:
    return OpenAIChatTranslator(config=config, model=model, base_url=base_url)
