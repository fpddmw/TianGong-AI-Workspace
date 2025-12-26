"""
Crossref API client focused on journal work listings.

The implementation wraps the `/journals/{issn}/works` endpoint exposed by
Crossref's public API and returns structured responses suitable for agent
consumption.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Sequence

import httpx

from .. import __version__

LOGGER = logging.getLogger(__name__)

__all__ = ["CrossrefClient", "CrossrefClientError"]

_SORT_ORDERS = {"asc", "desc"}


class CrossrefClientError(RuntimeError):
    """Raised when Crossref API requests fail or are misconfigured."""


@dataclass(slots=True)
class CrossrefClient:
    """Lightweight client for the Crossref Works API."""

    base_url: str = "https://api.crossref.org"
    timeout: float = 15.0
    mailto: str | None = None
    user_agent: str | None = None
    http_client: httpx.Client | None = None
    _default_headers: Mapping[str, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        agent = self.user_agent or f"TianGong-AI-Workspace/{__version__} (CrossrefClient)"
        headers: MutableMapping[str, str] = {
            "User-Agent": agent,
        }
        object.__setattr__(self, "_default_headers", headers)

    def list_journal_works(
        self,
        issn: str,
        *,
        query: str | None = None,
        filters: Mapping[str, Any] | Sequence[str] | str | None = None,
        sort: str | None = None,
        order: str | None = None,
        rows: int | None = None,
        offset: int | None = None,
        cursor: str | None = None,
        cursor_max: int | None = None,
        sample: int | None = None,
        select: Sequence[str] | str | None = None,
        mailto: str | None = None,
    ) -> Mapping[str, Any]:
        """
        Retrieve works for a specific journal by ISSN.

        Parameters mirror Crossref's `/journals/{issn}/works` endpoint. `filters`
        can be provided as either a pre-joined filter string (e.g.
        ``from-pub-date:2020-01-01,until-pub-date:2020-12-31``) or as a mapping
        which will be normalised into the expected `key:value` format.
        """

        issn_value = issn.strip()
        if not issn_value:
            raise CrossrefClientError("ISSN is required to query journal works.")

        params = _build_params(
            query=query,
            filters=filters,
            sort=sort,
            order=order,
            rows=rows,
            offset=offset,
            cursor=cursor,
            cursor_max=cursor_max,
            sample=sample,
            select=select,
            mailto=mailto or self.mailto,
        )

        url = f"{self.base_url.rstrip('/')}/journals/{issn_value}/works"
        try:
            response = self._get(url, params=params, headers=self._default_headers)
            response.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - defensive fallback
            LOGGER.exception("Crossref request failed")
            raise CrossrefClientError(f"HTTP error while querying Crossref: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:  # pragma: no cover - defensive fallback
            raise CrossrefClientError("Crossref returned invalid JSON.") from exc

        return {
            "issn": issn_value,
            "query": query,
            "parameters": params,
            "result": data,
        }

    def _get(self, url: str, *, params: Mapping[str, Any], headers: Mapping[str, str]) -> httpx.Response:
        if self.http_client is not None:
            return self.http_client.get(url, params=params, headers=headers, timeout=self.timeout)
        return httpx.get(url, params=params, headers=headers, timeout=self.timeout)


def _build_params(
    *,
    query: str | None,
    filters: Mapping[str, Any] | Sequence[str] | str | None,
    sort: str | None,
    order: str | None,
    rows: int | None,
    offset: int | None,
    cursor: str | None,
    cursor_max: int | None,
    sample: int | None,
    select: Sequence[str] | str | None,
    mailto: str | None,
) -> MutableMapping[str, Any]:
    params: MutableMapping[str, Any] = {}

    if query:
        params["query"] = query

    if filters:
        params["filter"] = _normalise_filters(filters)

    if sort:
        params["sort"] = sort

    if order:
        normalized_order = order.lower().strip()
        if normalized_order not in _SORT_ORDERS:
            raise CrossrefClientError("order must be either 'asc' or 'desc'.")
        params["order"] = normalized_order

    if rows is not None:
        if rows <= 0 or rows > 1000:
            raise CrossrefClientError("rows must be between 1 and 1000.")
        params["rows"] = int(rows)

    if offset is not None and cursor:
        raise CrossrefClientError("offset cannot be used together with cursor pagination.")
    if offset is not None:
        if offset < 0:
            raise CrossrefClientError("offset must be zero or positive.")
        params["offset"] = int(offset)

    if cursor:
        params["cursor"] = cursor
        if cursor_max is not None:
            if cursor_max < 0:
                raise CrossrefClientError("cursor_max must be zero or positive.")
            params["cursor_max"] = int(cursor_max)

    if sample is not None:
        if cursor:
            raise CrossrefClientError("sample cannot be combined with cursor-based pagination.")
        if sample <= 0:
            raise CrossrefClientError("sample must be greater than zero.")
        params["sample"] = int(sample)

    if select:
        params["select"] = _normalise_select(select)

    if mailto:
        params["mailto"] = mailto

    return params


def _normalise_filters(filters: Mapping[str, Any] | Sequence[str] | str) -> str:
    if isinstance(filters, str):
        cleaned = filters.strip()
        if not cleaned:
            raise CrossrefClientError("filter cannot be an empty string.")
        return cleaned

    if isinstance(filters, Mapping):
        parts = []
        for key, value in filters.items():
            key_str = str(key).strip()
            if not key_str:
                raise CrossrefClientError("filter keys cannot be empty.")
            if value is None:
                raise CrossrefClientError(f"filter '{key_str}' is missing a value.")
            parts.append(f"{key_str}:{value}")
        if not parts:
            raise CrossrefClientError("At least one filter must be provided.")
        return ",".join(parts)

    if isinstance(filters, Sequence):
        if not filters:
            raise CrossrefClientError("filters sequence cannot be empty.")
        parts = []
        for entry in filters:
            if not isinstance(entry, str):
                raise CrossrefClientError("filters sequence must contain only strings.")
            cleaned = entry.strip()
            if not cleaned:
                raise CrossrefClientError("filters cannot include empty strings.")
            parts.append(cleaned)
        return ",".join(parts)

    raise CrossrefClientError("filters must be a string, mapping, or sequence of filter expressions.")


def _normalise_select(value: Sequence[str] | str) -> str:
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            raise CrossrefClientError("select cannot be empty.")
        return cleaned

    fields = []
    for entry in value:
        cleaned = str(entry).strip()
        if not cleaned:
            raise CrossrefClientError("select cannot include empty field names.")
        fields.append(cleaned)
    if not fields:
        raise CrossrefClientError("select requires at least one field.")
    return ",".join(fields)
