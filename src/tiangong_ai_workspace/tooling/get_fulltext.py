"""Supabase sci_search client."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import httpx

from ..secrets import Secrets, SupabaseSecrets, load_secrets

__all__ = ["SupabaseClient", "SupabaseClientError"]


def _normalize_doi(value: str) -> str:
    """Strip common DOI URL/prefix forms and trailing punctuation."""

    raw = value.strip()
    lowered = raw.lower()
    prefixes = (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi.org/",
        "doi:",
    )
    for prefix in prefixes:
        if lowered.startswith(prefix):
            raw = raw[len(prefix) :]
            break
    return raw.strip(" .;,")


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

    def fetch_content(self, doi: str, *, top_k: int = 5, ext_k: int = 50) -> Mapping[str, Any]:
        if not doi.strip():
            raise SupabaseClientError("DOI cannot be empty.")
        normalized_doi = _normalize_doi(doi)
        payload = {
            "query": 'Introduction, Results, Discussions, Conclusions',
            "filter": {"doi": [normalized_doi]},
            "topK": top_k,
            "extK": ext_k,
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
