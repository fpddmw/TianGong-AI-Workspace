"""
Gemini Deep Research client helpers.

This module wraps the Gemini Interactions API to launch and monitor Deep
Research tasks. Calls are always performed with `background=True` and
`store=True` as required by the API.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Optional, Sequence

import httpx

from ..secrets import GeminiSecrets, Secrets, load_secrets

LOGGER = logging.getLogger(__name__)

__all__ = ["GeminiDeepResearchClient", "GeminiDeepResearchError"]


class GeminiDeepResearchError(RuntimeError):
    """Raised when the Gemini Interactions API returns an error."""


@dataclass(slots=True)
class GeminiDeepResearchClient:
    """Lightweight wrapper around the Gemini Deep Research agent."""

    secrets: Optional[Secrets] = None
    timeout: float = 30.0
    http_client: Optional[httpx.Client] = None
    _config: GeminiSecrets = field(init=False, repr=False)
    _base_url: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        loaded = self.secrets or load_secrets()
        config = loaded.gemini
        if config is None:
            raise GeminiDeepResearchError("Gemini credentials are not configured. Add a [gemini] section to `.sercrets/secrets.toml`.")

        base_url = (config.api_endpoint or "https://generativelanguage.googleapis.com").rstrip("/")
        object.__setattr__(self, "secrets", loaded)
        object.__setattr__(self, "_config", config)
        object.__setattr__(self, "_base_url", base_url)

    def start_research(
        self,
        prompt: str,
        *,
        agent: str | None = None,
        file_search_stores: Sequence[str] | None = None,
        include_thinking_summaries: bool = True,
    ) -> Mapping[str, Any]:
        """
        Start a Deep Research interaction in the background.

        The API requires both `background` and `store` to be true when running
        the Deep Research agent asynchronously.
        """

        if not prompt or not prompt.strip():
            raise GeminiDeepResearchError("Prompt cannot be empty.")

        payload: MutableMapping[str, Any] = {
            "input": prompt,
            "agent": agent or self._config.agent,
            "background": True,
            "store": True,
        }
        if file_search_stores:
            payload["tools"] = [
                {
                    "type": "file_search",
                    "file_search_store_names": list(file_search_stores),
                }
            ]
        if include_thinking_summaries:
            payload["agent_config"] = {
                "type": "deep-research",
                "thinking_summaries": "auto",
            }

        LOGGER.debug("Starting Gemini Deep Research interaction: %s", payload)
        response = self._post(self._interactions_url(), headers=self._headers(), json=payload)
        interaction = self._parse_json_response(response)
        return {
            "interaction": interaction,
            "interaction_id": interaction.get("id"),
            "status": interaction.get("status"),
        }

    def get_interaction(self, interaction_id: str) -> Mapping[str, Any]:
        """Fetch the latest state for a Deep Research interaction."""

        if not interaction_id or not interaction_id.strip():
            raise GeminiDeepResearchError("Interaction ID cannot be empty.")

        url = f"{self._interactions_url()}/{interaction_id}"
        response = self._get(url, headers=self._headers())
        interaction = self._parse_json_response(response)
        return {
            "interaction": interaction,
            "interaction_id": interaction.get("id") or interaction_id,
            "status": interaction.get("status"),
        }

    def poll_until_complete(
        self,
        interaction_id: str,
        *,
        interval: float = 10.0,
        max_attempts: int = 360,
    ) -> Mapping[str, Any]:
        """
        Poll an interaction until it completes or fails.

        Raises:
            GeminiDeepResearchError: if the interaction fails or exceeds max_attempts.
        """

        attempts = 0
        while True:
            attempts += 1
            interaction = self.get_interaction(interaction_id)
            status = str(interaction.get("status") or "").lower()
            if status == "completed":
                return interaction
            if status == "failed":
                error_detail = interaction.get("interaction", {}).get("error") or interaction.get("error") or {}
                raise GeminiDeepResearchError(f"Interaction {interaction_id} failed: {error_detail}")
            if attempts >= max_attempts:
                raise GeminiDeepResearchError(f"Interaction {interaction_id} did not complete after {attempts} polls.")
            time.sleep(interval)

    # ------------------------------------------------------------------ internals

    def _headers(self) -> Mapping[str, str]:
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": self._config.api_key,
        }

    def _interactions_url(self) -> str:
        return f"{self._base_url}/v1beta/interactions"

    def _parse_json_response(self, response: httpx.Response) -> Mapping[str, Any]:
        try:
            response.raise_for_status()
        except httpx.HTTPError as exc:
            LOGGER.exception("Gemini Deep Research request failed")
            raise GeminiDeepResearchError(f"HTTP error calling Gemini Interactions API: {exc}") from exc

        try:
            return response.json()
        except ValueError as exc:  # pragma: no cover - defensive fallback
            raise GeminiDeepResearchError("Gemini Interactions API returned invalid JSON.") from exc

    def _post(self, url: str, *, headers: Mapping[str, str], json: Mapping[str, Any]) -> httpx.Response:
        if self.http_client is not None:
            return self.http_client.post(url, headers=headers, json=json, timeout=self.timeout)
        return httpx.post(url, headers=headers, json=json, timeout=self.timeout)

    def _get(self, url: str, *, headers: Mapping[str, str]) -> httpx.Response:
        if self.http_client is not None:
            return self.http_client.get(url, headers=headers, timeout=self.timeout)
        return httpx.get(url, headers=headers, timeout=self.timeout)
