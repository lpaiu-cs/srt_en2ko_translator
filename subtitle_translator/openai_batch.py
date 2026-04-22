from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests


class OpenAIBatchClient:
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", timeout: int = 120):
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
        }

    def upload_batch_file(self, path: Path) -> dict:
        with path.open("rb") as handle:
            response = self.session.post(
                f"{self.base_url}/files",
                headers=self.headers,
                files={"file": (path.name, handle, "application/jsonl")},
                data={"purpose": "batch"},
                timeout=self.timeout,
            )
        response.raise_for_status()
        return response.json()

    def create_batch(
        self,
        input_file_id: str,
        endpoint: str = "/v1/chat/completions",
        completion_window: str = "24h",
        metadata: Optional[Dict[str, str]] = None,
    ) -> dict:
        response = self.session.post(
            f"{self.base_url}/batches",
            headers={**self.headers, "Content-Type": "application/json"},
            json={
                "input_file_id": input_file_id,
                "endpoint": endpoint,
                "completion_window": completion_window,
                "metadata": metadata or {},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def retrieve_batch(self, batch_id: str) -> dict:
        response = self.session.get(
            f"{self.base_url}/batches/{batch_id}",
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
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
        response = self.session.get(
            f"{self.base_url}/files/{file_id}/content",
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
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
