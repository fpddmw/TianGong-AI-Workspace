"""OpenAlex client and citation potential classifier."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

import httpx
from httpx import HTTPError
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI

from .llm import ModelRouter

__all__ = ["OpenAlexClient", "OpenAlexClientError", "LLMCitationAssessor"]

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

    def extract_pdf_url(self, work: Mapping[str, Any]) -> Optional[str]:
        """Return the best PDF URL available in a work record."""

        primary = work.get("primary_location") or {}
        if isinstance(primary, Mapping):
            pdf_url = primary.get("pdf_url")
            if isinstance(pdf_url, str) and pdf_url.startswith("http"):
                return pdf_url
        for loc in work.get("locations") or []:
            if not isinstance(loc, Mapping):
                continue
            pdf_url = loc.get("pdf_url")
            if isinstance(pdf_url, str) and pdf_url.startswith("http"):
                return pdf_url
        return None

    def download_pdf(self, url: str, dest: Path) -> Path:
        """Download a PDF to the destination path."""

        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with httpx.stream("GET", url, timeout=120.0) as response:
                response.raise_for_status()
                with dest.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        handle.write(chunk)
        except HTTPError as exc:
            raise OpenAlexClientError(f"Failed to download PDF from {url}: {exc}") from exc
        return dest

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


@dataclass(slots=True)
class LLMCitationAssessor:
    """LLM-driven classifier for article type and citation potential."""

    router: ModelRouter
    temperature: float = 0.2

    def assess(
        self,
        work: Mapping[str, Any],
        *,
        heuristic: Mapping[str, Any] | None = None,
        figure_notes: Optional[str] = None,
    ) -> Mapping[str, Any]:
        """Call the configured LLM to classify review vs research and rescore citation potential."""

        chat_model: ChatOpenAI = self.router.create_chat_model(
            purpose="general",
            temperature=self.temperature,
        )
        system_prompt = (
            "你是一名文献计量分析员，目标是根据论文的元数据（不包含下载/引用计数）、摘要要点与图表表现，对论文类型（综述/研究/其他）进行判断，并给出未来引用潜力评级。"
            "不要使用或推断具体的引用/下载数；请综合发表年份、新颖性、摘要信息密度、是否开放获取，以及图表对可读性的支持。"
            "输出 JSON，字段包括: article_type(综述|研究|其他), citation_category(high|medium|low), score(0-100), rationale(中文要点数组)。"
            "评分提示：综述类若覆盖面广、图表梳理清晰可倾向高分；研究类若方法/实验充分且图表解释力强、在同龄段具备可传播性则可加分。"
        )

        abstract_text = _flatten_abstract(work.get("abstract_inverted_index"))
        metadata = {
            "title": work.get("title"),
            "publication_year": work.get("publication_year"),
            "abstract_excerpt": abstract_text[:1200] if abstract_text else "",
            "type": work.get("type"),
            "open_access": bool(work.get("open_access", {}).get("is_oa")) if isinstance(work.get("open_access"), Mapping) else False,
        }

        user_payload: MutableMapping[str, Any] = {"metadata": metadata}
        if figure_notes:
            user_payload["figure_notes"] = figure_notes

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "请基于以下信息给出分类与引用潜力评分，输出 JSON 对象。",
                    },
                    {"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)},
                ]
            ),
        ]

        result: ChatResult = chat_model.invoke(messages, response_format={"type": "json_object"})
        content = result.generations[0].message.content if result.generations else result.content  # type: ignore[attr-defined]
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                raise OpenAlexClientError(f"LLM 返回的 JSON 无法解析: {exc}") from exc
        elif isinstance(content, Mapping):
            parsed = dict(content)
        else:  # pragma: no cover - defensive fallback
            raise OpenAlexClientError("LLM 返回的内容格式未知，期望 JSON 对象。")

        article_type = str(parsed.get("article_type") or "其他")
        citation_category = str(parsed.get("citation_category") or "medium").lower()
        score = float(parsed.get("score") or 50.0)
        rationale = parsed.get("rationale") or []
        if not isinstance(rationale, list):
            rationale = [str(rationale)]

        return {
            "article_type": article_type,
            "citation_category": citation_category,
            "score": score,
            "rationale": rationale,
            "raw": parsed,
        }
