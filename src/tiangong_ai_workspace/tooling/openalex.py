"""
OpenAlex client utilities for works metadata and cited-by queries.

This module mirrors the style of other tooling integrations (Crossref, Tavily)
so agents can reuse a consistent API surface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping

import httpx

LOGGER = logging.getLogger(__name__)

__all__ = ["OpenAlexClient", "OpenAlexClientError"]


class OpenAlexClientError(RuntimeError):
    """Raised when OpenAlex requests fail or are misconfigured."""


@dataclass(slots=True)
class OpenAlexClient:
    """Lightweight wrapper around the OpenAlex Works API."""

    base_url: str = "https://api.openalex.org"
    timeout: float = 15.0
    mailto: str | None = None
    http_client: httpx.Client | None = None

    def work_by_doi(self, doi: str, *, mailto: str | None = None) -> Mapping[str, Any]:
        """Fetch a work record using a DOI."""
        doi_value = doi.strip()
        if not doi_value:
            raise OpenAlexClientError("DOI is required.")
        url = f"{self.base_url.rstrip('/')}/works/https://doi.org/{doi_value}"
        params = _build_mailto_param(mailto or self.mailto)

        LOGGER.debug("Fetching OpenAlex work by DOI %s", doi_value)
        try:
            response = self._get(url, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - defensive fallback
            LOGGER.exception("OpenAlex work lookup failed")
            raise OpenAlexClientError(f"HTTP error while querying OpenAlex: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:  # pragma: no cover - defensive fallback
            raise OpenAlexClientError("OpenAlex returned invalid JSON.") from exc

        return {
            "doi": doi_value,
            "result": data,
        }

    def cited_by(
        self,
        work_id: str,
        *,
        from_publication_date: str | None = None,
        to_publication_date: str | None = None,
        per_page: int | None = 200,
        cursor: str | None = None,
        mailto: str | None = None,
    ) -> Mapping[str, Any]:
        """
        Retrieve works that cite the given OpenAlex work ID.

        The caller can filter citing works by publication date window.
        """

        work_value = work_id.strip()
        if not work_value:
            raise OpenAlexClientError("OpenAlex work ID is required.")

        params: MutableMapping[str, Any] = {"filter": f"cites:{work_value}"}

        if from_publication_date:
            params["filter"] += f",from_publication_date:{from_publication_date}"
        if to_publication_date:
            params["filter"] += f",to_publication_date:{to_publication_date}"
        if per_page is not None:
            if per_page <= 0 or per_page > 200:
                raise OpenAlexClientError("per_page must be between 1 and 200.")
            params["per-page"] = per_page
        if cursor:
            params["cursor"] = cursor

        params.update(_build_mailto_param(mailto or self.mailto))

        url = f"{self.base_url.rstrip('/')}/works"
        LOGGER.debug("Fetching OpenAlex cited-by for %s with params %s", work_value, params)
        try:
            response = self._get(url, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - defensive fallback
            LOGGER.exception("OpenAlex cited-by lookup failed")
            raise OpenAlexClientError(f"HTTP error while querying OpenAlex: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:  # pragma: no cover - defensive fallback
            raise OpenAlexClientError("OpenAlex returned invalid JSON.") from exc

        meta = data.get("meta") or {}
        count = meta.get("count")
        return {
            "work_id": work_value,
            "from_publication_date": from_publication_date,
            "to_publication_date": to_publication_date,
            "per_page": per_page,
            "cursor": cursor,
            "total_count": count,
            "result": data,
        }

    def _get(self, url: str, *, params: Mapping[str, Any]) -> httpx.Response:
        if self.http_client is not None:
            return self.http_client.get(url, params=params, timeout=self.timeout)
        return httpx.get(url, params=params, timeout=self.timeout)


def _build_mailto_param(mailto: str | None) -> MutableMapping[str, Any]:
    params: MutableMapping[str, Any] = {}
    if mailto:
        trimmed = mailto.strip()
        if trimmed:
            params["mailto"] = trimmed
    return params
