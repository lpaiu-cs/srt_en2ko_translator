from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple

from .config import RuntimeConfig
from .glossary import GlossaryStore
from .grouping import group_cues_into_sentences
from .models import Cue, SentenceGroup, TranslationRequest
from .splitting import groups_to_cues_by_time
from .text import normalize_text, warn
from .translators import BaseTranslator


def create_glossary_store(config: RuntimeConfig, glossary_log_path: Optional[str] = None) -> GlossaryStore:
    path = Path(glossary_log_path).expanduser() if glossary_log_path is not None else config.glossary_log_path
    return GlossaryStore(path=path, max_terms=config.glossary_max_terms)


def _last_nonempty_translation(groups: List[SentenceGroup], fallback: str) -> str:
    for group in reversed(groups):
        if group.text.strip():
            return group.text
    return fallback


def translate_batch_groups(
    batch_groups: List[SentenceGroup],
    translator: BaseTranslator,
    glossary_store: Optional[GlossaryStore],
    previous_translation: str = "",
    use_previous_context: bool = True,
) -> Tuple[List[SentenceGroup], str]:
    groups = deepcopy(batch_groups)
    count = len(batch_groups)
    if count == 0:
        return [], previous_translation

    texts = [group.text for group in batch_groups]
    if all(not text.strip() for text in texts):
        return groups, previous_translation

    request = TranslationRequest(
        sentences=texts,
        previous_translation=previous_translation if use_previous_context else "",
        glossary_terms=glossary_store.relevant_terms(texts) if glossary_store else [],
    )

    def fallback_single(text: str, prev_translation: str) -> str:
        single_request = TranslationRequest(
            sentences=[text],
            previous_translation=prev_translation if use_previous_context else "",
            glossary_terms=glossary_store.relevant_terms([text]) if glossary_store else [],
        )
        try:
            result = translator.translate_batch(single_request)
            if len(result.translations) != 1:
                raise ValueError(f"Expected 1 translation, got {len(result.translations)}")
            translated = normalize_text(result.translations[0])
            if text.strip() and not translated:
                raise ValueError("Translator returned empty text for non-empty input")
            if glossary_store:
                glossary_store.record_updates(result.glossary_updates)
            return translated
        except Exception as exc:
            warn(f"Falling back to source text for one sentence: {exc}")
            return text

    try:
        result = translator.translate_batch(request)
    except Exception:
        result = None

    if result is not None and len(result.translations) == count and not any("\n" in translation for translation in result.translations):
        normalized = [normalize_text(item) for item in result.translations]
        if not any(source.strip() and not translation for source, translation in zip(texts, normalized)):
            for group, translation in zip(groups, normalized):
                group.text = translation
            if glossary_store:
                glossary_store.record_updates(result.glossary_updates)
            return groups, _last_nonempty_translation(groups, previous_translation)

    if count == 1:
        groups[0].text = fallback_single(texts[0], previous_translation)
        return groups, _last_nonempty_translation(groups, previous_translation)

    midpoint = (count + 1) // 2
    left_groups, left_prev = translate_batch_groups(
        groups[:midpoint],
        translator=translator,
        glossary_store=glossary_store,
        previous_translation=previous_translation,
        use_previous_context=use_previous_context,
    )
    right_groups, right_prev = translate_batch_groups(
        groups[midpoint:],
        translator=translator,
        glossary_store=glossary_store,
        previous_translation=left_prev,
        use_previous_context=use_previous_context,
    )
    return left_groups + right_groups, right_prev


def translate_srt(
    cues: List[Cue],
    translator: BaseTranslator,
    batch_size: int = 32,
    repeat_fill: bool = False,
    glossary_store: Optional[GlossaryStore] = None,
    use_previous_context: bool = True,
) -> List[Cue]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    groups = group_cues_into_sentences(cues)
    translated_groups: List[SentenceGroup] = []
    previous_translation = ""
    for start in range(0, len(groups), batch_size):
        translated_batch, previous_translation = translate_batch_groups(
            groups[start : start + batch_size],
            translator=translator,
            glossary_store=glossary_store,
            previous_translation=previous_translation,
            use_previous_context=use_previous_context,
        )
        translated_groups.extend(translated_batch)
        time.sleep(0.2)
    return groups_to_cues_by_time(
        translated_groups,
        cues,
        repeat_fill=repeat_fill,
        width=42,
    )
