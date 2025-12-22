"""Supabase sci_search client."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import httpx

from ..secrets import Secrets, SupabaseSecrets, load_secrets

__all__ = ["SupabaseClient", "SupabaseClientError"]


class SupabaseClientError(RuntimeError):
    """Raised when Supabase sci_search requests fail."""


@dataclass(slots=True)
class SupabaseClient:
    """Minimal client for the sci_search edge function."""

    secrets: Optional[Secrets] = None
    timeout: float = 30.0
    http_client: Optional[httpx.Client] = None
    _config: SupabaseSecrets = field(init=False, repr=False)

    def __post_init__(self) -> None:
        loaded = self.secrets or load_secrets()
        config = loaded.supabase
        if config is None:
            raise SupabaseClientError("Supabase secrets are not configured.")
        object.__setattr__(self, "secrets", loaded)
        object.__setattr__(self, "_config", config)

    def fetch_content(self, doi: str, query: str, *, top_k: int = 5, est_k: int = 50) -> Mapping[str, Any]:
        if not doi.strip():
            raise SupabaseClientError("DOI cannot be empty.")
        payload = {
            "query": query,
            "filter": {"doi": [doi]},
            "topK": top_k,
            "estK": est_k,
        }
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._config.token,
            "x-region": "us-east-1",
        }
        try:
            response = self._post(self._config.api_url, json=payload, headers=headers)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SupabaseClientError(f"Supabase request failed: {exc}") from exc

        try:
            return response.json()
        except ValueError as exc:  # pragma: no cover - defensive
            raise SupabaseClientError("Supabase returned invalid JSON.") from exc

    def _post(self, url: str, *, json: Mapping[str, Any], headers: Mapping[str, str]) -> httpx.Response:
        if self.http_client is not None:
            return self.http_client.post(url, json=json, headers=headers, timeout=self.timeout)
        return httpx.post(url, json=json, headers=headers, timeout=self.timeout)
