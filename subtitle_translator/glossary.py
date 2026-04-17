from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .models import GlossaryEntry
from .text import normalize_text, warn


class GlossaryStore:
    def __init__(self, path: Optional[Path], max_terms: int = 12):
        self.path = path
        self.max_terms = max_terms
        self.entries: Dict[str, GlossaryEntry] = {}
        if self.path:
            self._load()

    @staticmethod
    def _key(source: str) -> str:
        return normalize_text(source).casefold()

    def _load(self) -> None:
        if not self.path or not self.path.exists():
            return
        for raw_line in self.path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
                entry = self._coerce_entry(payload)
                if entry:
                    self.entries[self._key(entry.source)] = entry
            except Exception as exc:
                warn(f"Skipping invalid glossary log line: {exc}")

    def _coerce_entry(self, payload: dict) -> Optional[GlossaryEntry]:
        source = normalize_text(str(payload.get("source", "")))
        target = normalize_text(str(payload.get("target", "")))
        note = normalize_text(str(payload.get("note", "")))
        mode = normalize_text(str(payload.get("mode", "soft"))).lower() or "soft"
        if not source or not target:
            return None
        if len(source) < 2 or len(source) > 80 or len(target) > 120:
            return None
        if source.isdigit():
            return None
        if mode not in {"hard", "soft"}:
            mode = "soft"
        return GlossaryEntry(source=source, target=target, note=note, mode=mode)

    def relevant_terms(self, texts: Iterable[str]) -> List[GlossaryEntry]:
        haystack = " ".join(texts)
        matched = [
            entry
            for entry in self.entries.values()
            if self._matches(entry.source, haystack)
        ]
        matched.sort(key=lambda entry: (-len(entry.source), entry.source.casefold()))
        return matched[: self.max_terms]

    def _matches(self, source: str, haystack: str) -> bool:
        normalized_source = normalize_text(source)
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 .+-/]*", normalized_source):
            pattern = rf"(?<!\w){re.escape(normalized_source)}(?!\w)"
            return re.search(pattern, haystack, flags=re.IGNORECASE) is not None
        return self._key(normalized_source) in haystack.casefold()

    def record_updates(self, updates: Iterable[GlossaryEntry]) -> None:
        if not self.path:
            return
        appended: List[dict] = []
        for update in updates:
            entry = self._coerce_entry(
                {
                    "source": update.source,
                    "target": update.target,
                    "note": update.note,
                    "mode": update.mode,
                }
            )
            if not entry:
                continue
            key = self._key(entry.source)
            current = self.entries.get(key)
            if current == entry:
                continue
            self.entries[key] = entry
            appended.append(
                {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "source": entry.source,
                        "target": entry.target,
                        "note": entry.note,
                        "mode": entry.mode,
                    }
                )
        if not appended:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for item in appended:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
