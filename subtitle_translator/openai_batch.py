from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests


class OpenAIBatchClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 120,
        max_attempts: int = 6,
        backoff_min_seconds: float = 1.0,
        backoff_max_seconds: float = 30.0,
    ):
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_attempts = max(1, max_attempts)
        self.backoff_min_seconds = max(0.1, backoff_min_seconds)
        self.backoff_max_seconds = max(self.backoff_min_seconds, backoff_max_seconds)
        self.session = requests.Session()

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
        }

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        last_exc: Exception | None = None
        url = f"{self.base_url}{path}"
        for attempt in range(self.max_attempts):
            try:
                response = self.session.request(method, url, timeout=self.timeout, **kwargs)
                if response.status_code in {429, 500, 502, 503, 504}:
                    retry_after = response.headers.get("retry-after")
                    base_delay = min(
                        self.backoff_max_seconds,
                        self.backoff_min_seconds * (2 ** attempt),
                    )
                    jitter_delay = random.uniform(
                        self.backoff_min_seconds,
                        max(self.backoff_min_seconds, base_delay),
                    )
                    delay = float(retry_after) if retry_after else jitter_delay
                    if attempt + 1 >= self.max_attempts:
                        response.raise_for_status()
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                return response
            except requests.HTTPError as exc:
                last_exc = exc
                raise
            except requests.RequestException as exc:
                last_exc = exc
                if attempt + 1 >= self.max_attempts:
                    raise
                base_delay = min(
                    self.backoff_max_seconds,
                    self.backoff_min_seconds * (2 ** attempt),
                )
                time.sleep(
                    random.uniform(
                        self.backoff_min_seconds,
                        max(self.backoff_min_seconds, base_delay),
                    )
                )
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Batch API request failed without a concrete exception.")

    def upload_batch_file(self, path: Path) -> dict:
        with path.open("rb") as handle:
            response = self._request(
                "POST",
                "/files",
                headers=self.headers,
                files={"file": (path.name, handle, "application/jsonl")},
                data={"purpose": "batch"},
            )
        return response.json()

    def create_batch(
        self,
        input_file_id: str,
        endpoint: str = "/v1/chat/completions",
        completion_window: str = "24h",
        metadata: Optional[Dict[str, str]] = None,
    ) -> dict:
        response = self._request(
            "POST",
            "/batches",
            headers={**self.headers, "Content-Type": "application/json"},
            json={
                "input_file_id": input_file_id,
                "endpoint": endpoint,
                "completion_window": completion_window,
                "metadata": metadata or {},
            },
        )
        return response.json()

    def retrieve_batch(self, batch_id: str) -> dict:
        response = self._request(
            "GET",
            f"/batches/{batch_id}",
            headers=self.headers,
        )
        return response.json()

    def wait_for_batch(self, batch_id: str, poll_interval_seconds: int = 30, timeout_seconds: int = 60 * 60 * 24) -> dict:
        started = time.monotonic()
        while True:
            batch = self.retrieve_batch(batch_id)
            status = batch.get("status")
            if status in {"completed", "failed", "cancelled", "expired"}:
                return batch
            if time.monotonic() - started > timeout_seconds:
                raise TimeoutError(f"Timed out waiting for batch {batch_id}")
            time.sleep(poll_interval_seconds)

    def download_file_text(self, file_id: str) -> str:
        response = self._request(
            "GET",
            f"/files/{file_id}/content",
            headers=self.headers,
        )
        return response.text

    def download_file_to_path(self, file_id: str, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.download_file_text(file_id), encoding="utf-8")
        return output_path


def parse_batch_output_line(raw_line: str) -> Dict[str, Any]:
    line = raw_line.strip()
    if not line:
        return {}
    return json.loads(line)
