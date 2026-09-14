from __future__ import annotations

import requests


def retryable_response(response: requests.Response) -> bool:
    if response.status_code == 429:
        try:
            error = response.json().get("error", {})
            if isinstance(error, dict) and any(
                error.get(key) in {"insufficient_quota", "credit_balance_exhausted"}
                for key in ("type", "code")
            ):
                return False
        except (ValueError, AttributeError):
            pass
        return True
    return response.status_code in {408, 500, 502, 503, 504}
