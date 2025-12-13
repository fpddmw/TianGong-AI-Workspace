"""OpenAlex client and citation potential classifier."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

import httpx

__all__ = ["OpenAlexClient", "OpenAlexClientError"]

BASE_URL = "https://api.openalex.org"


class OpenAlexClientError(RuntimeError):
    """Raised when OpenAlex API calls fail."""


def _flatten_abstract(inverted_index: Mapping[str, Iterable[int]] | None) -> str:
    if not inverted_index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, indexes in inverted_index.items():
        for idx in indexes:
            positions.append((idx, word))
    return " ".join(word for _, word in sorted(positions, key=lambda pair: pair[0]))


@dataclass(slots=True)
class OpenAlexClient:
    """Thin wrapper around the OpenAlex works API plus a heuristic classifier."""

    base_url: str = BASE_URL
    email: Optional[str] = None
    timeout: float = 20.0
    http_client: Optional[httpx.Client] = None

    def search_works(
        self,
        query: str,
        *,
        since_year: int | None = None,
        per_page: int = 20,
        sample: bool = False,
        sort: str = "cited_by_count:desc",
        fields: Optional[Sequence[str]] = None,
    ) -> list[Mapping[str, Any]]:
        """Search for works using OpenAlex."""

        if not query.strip():
            raise OpenAlexClientError("Query cannot be empty.")

        params: MutableMapping[str, Any] = {
            "search": query,
            "per-page": per_page,
            "sort": sort,
        }
        if self.email:
            params["mailto"] = self.email
        if since_year:
            params["filter"] = f"from_publication_date:{since_year}-01-01"
        if sample:
            params["sample"] = per_page
        if fields:
            params["select"] = ",".join(fields)

        url = f"{self.base_url}/works"
        try:
            response = self._get(url, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OpenAlexClientError(f"OpenAlex request failed: {exc}") from exc

        try:
            payload = response.json()
        except ValueError as exc:  # pragma: no cover - defensive fallback
            raise OpenAlexClientError("OpenAlex returned invalid JSON.") from exc

        results = payload.get("results")
        if not isinstance(results, list):
            raise OpenAlexClientError("OpenAlex response missing 'results'.")
        return results

    def classify_work(self, work: Mapping[str, Any]) -> Mapping[str, Any]:
        """Assign a coarse citation potential category with rationale."""

        cited_by = int(work.get("cited_by_count") or 0)
        year = int(work.get("publication_year") or 0)
        reference_count = int(work.get("referenced_works_count") or 0)
        abstract_text = _flatten_abstract(work.get("abstract_inverted_index"))
        abstract_words = len(abstract_text.split()) if abstract_text else 0
        is_oa = bool(work.get("open_access", {}).get("is_oa")) if isinstance(work.get("open_access"), Mapping) else False

        score = 0.0
        rationale: list[str] = []

        # Citations carry the most weight.
        if cited_by >= 200:
            score += 3
            rationale.append("高引用次数(≥200)")
        elif cited_by >= 80:
            score += 2.5
            rationale.append("较高引用次数(≥80)")
        elif cited_by >= 20:
            score += 1
            rationale.append("中等引用次数(≥20)")
        else:
            rationale.append("引用次数较低(<20)")

        # Recency: newer work has less time to accumulate citations.
        current_year = datetime.now(timezone.utc).year
        if year >= current_year - 2:
            score += 0.3
            rationale.append("近发表年份，未来潜力待观察")
        elif year <= current_year - 7:
            score += 0.2
            rationale.append("发表时间较久，引用积累充分")

        # Abstract depth as a proxy for clarity/structure.
        if abstract_words >= 200:
            score += 0.5
            rationale.append("摘要较长，信息密度高")
        elif abstract_words < 50:
            score -= 0.2
            rationale.append("摘要较短，信息有限")

        # Reference breadth.
        if reference_count >= 30:
            score += 0.3
            rationale.append("引用文献丰富(≥30)")
        elif reference_count == 0:
            score -= 0.2
            rationale.append("缺少参考文献计数")

        # Accessibility proxy.
        if is_oa:
            score += 0.2
            rationale.append("开放获取可见度更高")

        if score >= 3.0:
            category = "high"
        elif score >= 1.8:
            category = "medium"
        else:
            category = "low"

        return {
            "id": work.get("id"),
            "title": work.get("title"),
            "publication_year": year or None,
            "cited_by_count": cited_by,
            "reference_count": reference_count,
            "open_access": is_oa,
            "abstract_words": abstract_words,
            "category": category,
            "score": round(score, 2),
            "rationale": rationale,
        }

    def _get(self, url: str, params: Mapping[str, Any]) -> httpx.Response:
        if self.http_client is not None:
            return self.http_client.get(url, params=params, timeout=self.timeout)
        return httpx.get(url, params=params, timeout=self.timeout)
