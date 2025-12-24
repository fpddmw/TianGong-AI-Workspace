"""OpenAlex client and citation potential classifier."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

import httpx
from httpx import HTTPError
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pypdf import PdfReader

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

    def search_by_doi(self, doi: str) -> Mapping[str, Any]:
        """Lookup a single work by DOI."""

        if not doi.strip():
            raise OpenAlexClientError("DOI cannot be empty.")
        params: MutableMapping[str, Any] = {"filter": f"doi:{doi}"}
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
        results = payload.get("results") or []
        if not results:
            raise OpenAlexClientError(f"No OpenAlex record found for DOI '{doi}'.")
        return results[0]

    def extract_doi_from_pdf(self, pdf_path: Path) -> str:
        """Heuristically extract a DOI from a PDF's text."""

        if not pdf_path.exists() or not pdf_path.is_file():
            raise OpenAlexClientError(f"PDF not found: {pdf_path}")
        try:
            reader = PdfReader(str(pdf_path))
        except Exception as exc:  # pragma: no cover - defensive
            raise OpenAlexClientError(f"Failed to read PDF {pdf_path}: {exc}") from exc

        doi_pattern = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)
        for page in reader.pages:
            try:
                text = page.extract_text() or ""
            except Exception:
                continue
            match = doi_pattern.search(text)
            if match:
                return match.group(0)
        raise OpenAlexClientError("DOI not found in PDF.")

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
        fulltext: Optional[str] = None,
        fulltext_source: Optional[str] = None,
        figure_notes: Optional[str] = None,
    ) -> Mapping[str, Any]:
        """Call the configured LLM to score citation potential with RCR-specific rubric."""

        chat_model: ChatOpenAI = self.router.create_chat_model(
            purpose="general",
            temperature=self.temperature,
        )
        system_prompt_parts = [
            "Role 你是一位深耕于《Resources, Conservation & Recycling》(RCR) 期刊的资深审稿专家与学术计量学分析师,能够通过分析文章的摘要、方法论和数据特征，准确预判其两年后的引用水平并提供针对性的修改建议。",
            "Goal 根据用户提供的论文初稿信息（标题、摘要、方法、数据描述），基于“RCR 核心规则库”进行多维度打分，预测引文表现，并给出旨在提升“引文潜力”的改进方案。",
            "RCR Core Rulebook (核心规则库)",
            "1. 选题战略维度 (Topic Strategic Fit)",
            " - 【规则 1.1：跨学科耦合】高引论文必须关联两个以上的前沿领域。核心关键词：AI/机器学习、ESG 风险、全球能源转型、供应链安全、碳中和路径、关键金属安全。",
            " - 【规则 1.2：尺度溢价】全球(Global) > 国家(National) > 跨区域 > 单一城市 > 单一工厂。",
            " - 【规则 1.3：通用性陷阱】若研究对象为特定地区，必须在方法论上提出“可迁移框架”或“普适性机理”，否则视为 Low Band 倾向。",
            "2. 方法论严谨性维度 (Methodological Rigor)",
            " - 【规则 2.1：闭环评价原则】材料/工程类文章必须包含“结构设计-性能评估-LCA(环境)-LCC(经济)”的完整闭环。",
            " - 【规则 2.2：算法壁垒】涉及 AI/ML 时，必须包含：5种以上算法对比、嵌套交叉验证、模型可解释性分析（如 SHAP 值）。",
            " - 【规则 2.3：定量化门槛】定性研究（访谈/政策评论）若无系统编码或情景模拟（如系统动力学/SD），被引潜力通常极低。",
            "3. 数据与证据维度 (Data & Evidence Utility)",
            " - 【规则 3.1：数据权威性】优先使用 UN Comtrade, USGS, IEA, S&P Global 等权威二阶数据，或大规模卫星遥感数据。",
            " - 【规则 3.2：时效性红线】政策分析类数据滞后不得超过 3 年。",
            " - 【规则 3.3：可视化标准】鼓励使用高质量 Sankey 图（物质流）、GIS 热力图和复杂的关联网络图。",
            "4. 影响力与决策支撑 (Impact & Decision Support)",
            " - 【规则 4.1：政策锚点】结论必须直接呼应具体国际条约或政策框架（如欧盟绿色协议、巴塞尔公约等）。",
            " - 【规则 4.2：行动导向】拒绝“加强教育”等空洞建议，必须提供量化的、可操作的政策参数或改进点。",
            "Evaluation Logic (评估逻辑)",
            " - High Band：在 1.1, 1.2, 2.1, 4.1 规则中至少有三项表现卓越（优）。",
            " - Middle Band：表现规范，但缺乏全球尺度或方法论创新较弱。",
            " - Low Band：选题过窄、方法陈旧、或数据时效性差。",
            "Constraints",
            "评价需尖锐且客观，不要使用模棱两可的学术辞令。",
            "必须直接指出违反了哪条 RCR Core Rulebook 中的具体规则。",
            "语言统一使用中文。",
        ]
        if figure_notes:
            system_prompt_parts.append(
                "附加要求：已提供图表拆解，请结合图表所揭示的流程/数据/实验设计判断其是否支撑论文结论，"
                "并在方法、数据或影响力维度中体现图表的解释力或缺陷。"
            )
        system_prompt = "\n".join(system_prompt_parts)

        response_schema: Mapping[str, Any] = {
            "name": "rcr_citation_assessment",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "prediction": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "estimated_band": {"type": "string", "enum": ["High", "Middle", "Low"]},
                            "confidence_score": {"type": "string", "pattern": r"^\d{1,3}%$"},
                            "key_reason": {"type": "string"},
                        },
                        "required": ["estimated_band", "confidence_score", "key_reason"],
                    },
                    "dimension_scores": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "topic": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "score": {"type": "integer", "minimum": 1, "maximum": 3},
                                    "eval": {"type": "string", "enum": ["优", "中", "差"]},
                                    "analysis": {"type": "string"},
                                },
                                "required": ["score", "eval", "analysis"],
                            },
                            "methodology": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "score": {"type": "integer", "minimum": 1, "maximum": 3},
                                    "eval": {"type": "string", "enum": ["优", "中", "差"]},
                                    "analysis": {"type": "string"},
                                },
                                "required": ["score", "eval", "analysis"],
                            },
                            "data": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "score": {"type": "integer", "minimum": 1, "maximum": 3},
                                    "eval": {"type": "string", "enum": ["优", "中", "差"]},
                                    "analysis": {"type": "string"},
                                },
                                "required": ["score", "eval", "analysis"],
                            },
                            "impact": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "score": {"type": "integer", "minimum": 1, "maximum": 3},
                                    "eval": {"type": "string", "enum": ["优", "中", "差"]},
                                    "analysis": {"type": "string"},
                                },
                                "required": ["score", "eval", "analysis"],
                            },
                        },
                        "required": ["topic", "methodology", "data", "impact"],
                    },
                    "action_plan": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "rcr_match_index": {"type": "string", "pattern": r"^\d{1,3}%$"},
                },
                "required": ["prediction", "dimension_scores", "action_plan", "rcr_match_index"],
            },
            "strict": True,
        }

        abstract_text = _flatten_abstract(work.get("abstract_inverted_index"))
        metadata = {
            "title": work.get("title"),
            "publication_year": work.get("publication_year"),
            "abstract_excerpt": abstract_text[:1200] if abstract_text else "",
            "type": work.get("type"),
            "open_access": bool(work.get("open_access", {}).get("is_oa")) if isinstance(work.get("open_access"), Mapping) else False,
        }

        user_payload: MutableMapping[str, Any] = {
            "metadata": metadata,
            "fulltext_source": fulltext_source or "unspecified",
        }
        if fulltext:
            user_payload["fulltext_excerpt"] = fulltext
        if figure_notes:
            user_payload["figure_notes"] = figure_notes
        if heuristic:
            user_payload["baseline"] = heuristic

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

        raw = chat_model.invoke(messages, response_format={"type": "json_schema", "json_schema": response_schema})
        content: Any
        parsed: Any = None
        if hasattr(raw, "parsed") and getattr(raw, "parsed") is not None:  # OpenAI JSON schema response
            parsed = getattr(raw, "parsed")
        if parsed is None:
            if hasattr(raw, "generations"):  # ChatResult
                result = raw  # type: ignore[assignment]
                content = result.generations[0].message.content if result.generations else result.content  # type: ignore[attr-defined]
            elif hasattr(raw, "content"):  # AIMessage or similar
                content = getattr(raw, "content")
            else:  # pragma: no cover - defensive
                content = raw

            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError as exc:
                    raise OpenAlexClientError(f"LLM 返回的 JSON 无法解析: {exc}") from exc
            elif isinstance(content, Mapping):
                parsed = dict(content)
            else:  # pragma: no cover - defensive fallback
                raise OpenAlexClientError("LLM 返回的内容格式未知，期望 JSON 对象。")

        if not isinstance(parsed, Mapping):
            raise OpenAlexClientError("LLM 返回的内容格式未知，期望 JSON 对象。")

        return {
            "prediction": parsed.get("prediction"),
            "dimension_scores": parsed.get("dimension_scores"),
            "action_plan": parsed.get("action_plan"),
            "rcr_match_index": parsed.get("rcr_match_index"),
            "raw": parsed,
        }
