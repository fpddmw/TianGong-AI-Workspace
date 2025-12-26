"""OpenAlex client and citation potential classifier."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
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
            (
                "Role\n"
                "你是一位在环境科学、资源循环与可持续发展领域极具影响力的学术战略顾问。你擅长通过深度剖析论文的内在逻辑、方法严谨性与数据规模，"
                "精准预判一篇稿件在期刊中的学术表现，并能给出极具建设性的修改意见。"
            ),
            ("Goal\n" "根据用户提供的论文初稿信息（标题、摘要、方法、数据描述），基于“RCR 核心规则库”进行多维度打分，预测引文表现，" "并给出旨在提升“引文潜力”的改进方案。"),
            "Evaluation Criteria",
            (
                "请基于以下四个深度维度对用户提供的稿件信息进行评估：\n"
                "1. 选题的系统性与全球视野\n"
                "1.1跨领域耦合性：顶尖研究不再孤立地讨论“资源回收”，而是将其嵌入宏观系统中。核心在于是否能将AI/机器学习、循环经济与能源转型、地缘政治、供应链安全、碳中和路径或ESG 风险等全球大议题进行深度耦合。\n"
                "1.2尺度溢价：评估研究是否具备“尺度跨越”能力。相比于单一工厂或特定城市的个案研究，具备全球视角（Global）、国家尺度（National）或跨区域流动分析的研究具有更高的学术价值。\n"
                "1.3普适性范式：如果研究涉及特定地区，需审查其是否提炼出了可迁移的方法论框架。若仅停留于“本地现状描述”而无理论外延，其学术影响力将严重受限。\n"
                "2. 方法论的集成深度与逻辑闭环\n"
                "2.1全生命周期闭环：在资源利用研究中，优秀的论文应构建“机理探索—性能验证—环境影响（LCA）—经济可行性（LCC）”的闭环。缺乏环境效益或经济可行性对冲的技术研究往往显得单薄。\n"
                "2.2计算与算法壁垒：若涉及数据建模或机器学习，需达到极高的工业严谨度。包括但不限于：多算法的横向对比验证、模型的可解释性分析（如 SHAP 值）、以及针对类不平衡或噪声数据的特殊鲁棒性处理。\n"
                "2.3量化模拟深度：对于政策类文章，应超越纯定性的文本梳理。通过系统动力学（SD）、情景模拟或多代理模型（ABM）进行的量化预测，是区分普通综述与顶级研究的关键。\n"
                "3. 数据效力与可视化表达\n"
                "3.1数据权威性与规模：优先评估是否使用了权威的一阶/二阶数据源（如 S&P Global, IEA, UN Comtrade 等）。数据量级应能支撑起时间维度的趋势预测或空间维度的分布推演。\n"
                "3.2时效性红线：环境政策与技术发展日新月异，数据来源若滞后于当前时间点 3 年以上，通常会被认为缺乏现实指导意义。\n"
                "3.3专业可视化标准：高水平论文通常配有极具信息密度的可视化图表，如：展示物质流流向的桑基图、揭示地理分布的热力图以及展示变量间因果关联的结构方程模型图。\n"
                "4. 决策支持与行动导向\n"
                "4.1政策锚定精准度：结论是否直接回应了具体的国际协议、政府规划或行业标准。拒绝空洞的建议，优秀的论文应给出具体的参数（如：建议某项税收上调多少百分比能实现最佳回收率）。\n"
                "4.2管理启示的深度：评估文章是否为决策者（政府或企业管理层）提供了具有“干预价值”的洞察，而非单纯的学术发现总结。"
            ),
            (
                "Task\n"
                "请针对用户提供的论文信息，对比上述标准完成："
                "详细说明该文章在上述四个维度的表现。指出该文章目前最急需解决的一个致命弱点。给出具体的改进建议，帮助作者将文章提升至期刊录用标准。"
                "定性描述该文章目前处于该领域中的位置（领先/中等/滞后），并预判其可能的被引潜力。"
            ),
            "Evaluation Logic",
            "领先水平：在1.1, 1.2, 2.1, 4.1 准则中至少有三项表现卓越（优）。",
            "中等水平：表现规范，但缺乏全球尺度或方法论创新较弱。",
            "落后水平：选题过窄、方法陈旧、或数据时效性差。",
            ("Constraints\n" "语言风格应尖锐且严肃、客观、专业，避免使用模板化的评价词汇，必须结合用户提供的具体研究内容进行深度点评。" "语言统一使用中文。"),
        ]
        if figure_notes:
            system_prompt_parts.append("附加要求：已提供图表拆解，请结合图表所揭示的流程/数据/实验设计判断其是否支撑论文结论，并在方法、数据或影响力维度中体现图表的解释力或缺陷。")
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
                            "key_reason": {"type": "string"},
                        },
                        "required": ["estimated_band", "key_reason"],
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
                },
                "required": ["prediction", "dimension_scores", "action_plan"],
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
            "raw": parsed,
        }
