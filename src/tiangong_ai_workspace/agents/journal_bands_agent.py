"""Agent dedicated to journal citation band analysis."""

from __future__ import annotations

import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from ..tooling.get_fulltext import SupabaseClient, SupabaseClientError
from ..tooling.highly_cited import fetch_journal_citation_bands
from ..tooling.llm import ModelRouter

__all__ = ["JournalBandsConfig", "JournalBandsAgent", "run_journal_bands_agent"]


@dataclass(slots=True)
class JournalBandsConfig:
    journal_issn: str
    journal_name: str
    publish_year: Optional[int] = None
    band_limit: int = 200
    max_records: int = 800
    top_k: int = 10
    ext_k: int = 80
    show_fulltext: bool = False
    output: Path = Path("journal_bands_summary.md")


@dataclass(slots=True)
class JournalBandsAgent:
    """Run Supabase+LLM sampling for journal citation bands."""

    supabase: SupabaseClient | None = None
    router: ModelRouter | None = None
    max_workers: int = 10
    response_temperature: float = 0.2

    def __post_init__(self) -> None:
        if self.supabase is None:
            self.supabase = SupabaseClient()
        if self.router is None:
            self.router = ModelRouter()

    def run(self, config: JournalBandsConfig, *, log: Callable[[str], None] | None = None) -> Path:
        def _log(msg: str) -> None:
            if log:
                log(msg)

        _log("Fetching citation bands from OpenAlex...")
        bands = fetch_journal_citation_bands(
            journal_issn=config.journal_issn,
            journal_name=config.journal_name,
            publish_year=config.publish_year,
            band_limit=config.band_limit,
            max_records=config.max_records,
        )
        _log(
            f"Fetched bands: high={len(bands.high)}, middle={len(bands.middle)}, low={len(bands.low)} "
            f"(p25={bands.thresholds.get('p25', 0):.1f}, p75={bands.thresholds.get('p75', 0):.1f})"
        )

        assert self.supabase is not None  # for type-checkers
        assert self.router is not None
        chat_model = self.router.create_chat_model(purpose="general", temperature=self.response_temperature)

        response_schema = self._build_schema()
        sections: list[tuple[str, list[Mapping[str, Any]]]] = [
            ("High citation (top)", bands.high),
            ("Middle citation (mid band)", bands.middle),
            ("Low citation (bottom)", bands.low),
        ]

        output_path = config.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        header_lines: list[str] = []
        header_lines.append(f"# Journal analysis: {config.journal_name} (ISSN {config.journal_issn})")
        if config.publish_year:
            header_lines.append(f"- Publication year filter: {config.publish_year}")
        header_lines.append(
            f"- Band limit: {config.band_limit} (after percentile split), max records considered: {config.max_records}"
        )
        header_lines.append(
            f"- Citation thresholds: p25={bands.thresholds.get('p25', 0):.1f}, p75={bands.thresholds.get('p75', 0):.1f}"
        )
        header_lines.append(f"- Supabase sci_search: topK={config.top_k}, extK={config.ext_k}")
        header_lines.append("")
        output_path.write_text("\n".join(header_lines) + "\n", encoding="utf-8")

        for section_title, works in sections:
            self._write_section_header(output_path, section_title, works)
            if not works:
                continue
            sampled_works = self._sample_band(works)
            _log(
                f"Processing {len(sampled_works)} sampled papers (source pool: {len(works)}) in {section_title}..."
            )

            tasks = list(enumerate(sampled_works, start=1))
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(
                    executor.map(
                        lambda task: self._analyze_work(
                            task,
                            config=config,
                            section_title=section_title,
                            response_schema=response_schema,
                            chat_model=chat_model,
                        ),
                        tasks,
                    )
                )

            for idx, entry_lines in sorted(results, key=lambda item: item[0]):
                with output_path.open("a", encoding="utf-8") as handle:
                    handle.write("\n".join(entry_lines))
                if idx % 5 == 0 or idx == len(sampled_works):
                    _log(f"{section_title}: {idx}/{len(sampled_works)} written")

        _log(f"Summary streaming written to {output_path}")
        return output_path

    def _build_schema(self) -> Mapping[str, Any]:
        feature_eval_schema: Mapping[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "eval": {"type": "string", "enum": ["优", "中", "差"]},
                "reason": {"type": "string"},
            },
            "required": ["eval", "reason"],
        }
        return {
            "name": "journal_band_assessment",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "citation_band": {"type": "string", "enum": ["High", "Middle", "Low"]},
                    "article_type": {"type": "string", "enum": ["研究", "综述", "其他"]},
                    "features_analysis": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "topic": feature_eval_schema,
                            "methodology": feature_eval_schema,
                            "data": feature_eval_schema,
                            "impact": feature_eval_schema,
                            "presentation": feature_eval_schema,
                        },
                        "required": ["topic", "methodology", "data", "impact", "presentation"],
                    },
                    "rule_suggestion": {"type": "string"},
                },
                "required": ["citation_band", "article_type", "features_analysis", "rule_suggestion"],
            },
            "strict": True,
        }

    def _sample_band(self, items: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        if not items:
            return []
        sample_size = max(1, int(math.ceil(len(items) * 0.1)))
        return random.sample(items, min(sample_size, len(items)))

    def _write_section_header(self, output_path: Path, section_title: str, works: list[Mapping[str, Any]]) -> None:
        section_lines: list[str] = [f"## {section_title}"]
        if not works:
            section_lines.append("_No records found._")
            section_lines.append("")
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(section_lines))
            return

        section_lines.append(f"_Sampled {max(1, int(math.ceil(len(works) * 0.1)))} of {len(works)} papers (10% random)._")
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(section_lines) + "\n")

    def _analyze_work(
        self,
        task: tuple[int, Mapping[str, Any]],
        *,
        config: JournalBandsConfig,
        section_title: str,
        response_schema: Mapping[str, Any],
        chat_model: Any,
    ) -> tuple[int, list[str]]:
        idx, work = task
        doi = work.get("doi")
        title = work.get("title") or "(untitled)"
        citation_count = work.get("cited_by_count")
        year = work.get("publication_year")
        try:
            fulltext = self.supabase.fetch_content(
                str(doi or ""),
                top_k=config.top_k,
                ext_k=config.ext_k,
            )
        except SupabaseClientError as exc:
            rationale = f"Supabase 获取全文失败: {exc}"
            entry_lines = [
                f"- **{title}** ({year}, cites: {citation_count}) DOI: {doi or 'n/a'}",
                f"  - {rationale}",
                "",
            ]
            return idx, entry_lines

        content_text = json.dumps(fulltext, ensure_ascii=False)
        system_prompt = (
            "Role\n"
            "你是一位资深的学术计量学专家与期刊审稿人。你的任务是分析学术论文的特征，并探讨这些特征与其引文表现，即引用数及其引用带（High/Middle/Low）之间的潜在逻辑关系，为构建“投稿指导规则库”提供结构化素材。"
            "Task"
            "请根据待分析内容，从以下五个维度对文章进行深度特征分析，并输出 JSON 格式的评价："
            "选题前沿性：是否涉及热点、新兴领域或交叉学科。"
            "方法论完备性：实验设计、模型创新度、逻辑严密程度。"
            "数据/证据效力：样本规模、数据来源的权威性、实证分析的深度。"
            "结论影响深度：是否提供了决策支持、政策建议或理论突破。"
            "表述与规范性：论证逻辑、专业术语使用的准确性。"
            "Constraints"
            "严禁捏造数据。"
            "评价需客观、简练，必须直接指出该维度下的“具体特征点”。"
            "语言统一使用中文。"
            "输出必须是合法的 JSON 对象，字段需严格匹配预定义 schema。"
            "明确区分文章类型（学术研究/综述/其他），并给出理由。"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    "待分析内容\n"
                    f"- 引用表现: 该期刊中引用排名位于{section_title}，引用量为{citation_count}\n"
                    f"- 标题: {title}\n"
                    f"- 作者: {work.get('authorships') or '未知'}\n"
                    f"- 摘要/片段: {content_text}\n"
                    f"请基于以上信息，分析为何该文章会拥有这样的引用表现，并按照要求的 JSON 格式输出。"
                )
            ),
        ]
        summary: Mapping[str, Any] | None = None
        raw_text: str | None = None
        try:
            resp = chat_model.invoke(messages, response_format={"type": "json_schema", "json_schema": response_schema})
            content = getattr(resp, "content", None) or (resp.generations[0].message.content if hasattr(resp, "generations") else None)
            if isinstance(content, str):
                raw_text = content
                try:
                    summary = json.loads(content)
                except json.JSONDecodeError:
                    summary = None
            elif isinstance(content, Mapping):
                summary = dict(content)
                raw_text = json.dumps(summary, ensure_ascii=False)
            else:
                raw_text = str(content)
        except Exception as exc:  # pragma: no cover - defensive
            raw_text = f"LLM 调用失败: {exc}"

        entry_lines = [f"- **{title}** ({year}, cites: {citation_count}) DOI: {doi or 'n/a'}"]
        if config.show_fulltext:
            entry_lines.append(f"  - Fulltext excerpt (len={len(content_text)}): {content_text}")
            if len(content_text.strip()) < 10:
                entry_lines.append(f"  - Supabase raw payload: {json.dumps(fulltext, ensure_ascii=False)[:400]}")
        if isinstance(summary, Mapping):
            article_type = summary.get("article_type")
            if article_type:
                entry_lines.append(f"  - 文章类型: {article_type}")
            strengths = summary.get("strengths") or summary.get("pros") or summary.get("advantages")
            weaknesses = summary.get("weaknesses") or summary.get("cons") or summary.get("risks")
            if strengths:
                if isinstance(strengths, str):
                    strengths = [strengths]
                entry_lines.append("  - 优势:")
                for item in strengths:
                    entry_lines.append(f"    - {item}")
            if weaknesses:
                if isinstance(weaknesses, str):
                    weaknesses = [weaknesses]
                entry_lines.append("  - 不足:")
                for item in weaknesses:
                    entry_lines.append(f"    - {item}")
        else:
            entry_lines.append("  - 无法解析 LLM 输出，原始响应略。")
        if raw_text:
            entry_lines.append("  - 原始 LLM 输出:")
            entry_lines.append(f"    - {raw_text}")
        entry_lines.append("")
        return idx, entry_lines


def run_journal_bands_agent(config: JournalBandsConfig, *, log: Callable[[str], None] | None = None) -> Path:
    """Convenience wrapper for running the journal bands agent."""

    agent = JournalBandsAgent()
    return agent.run(config, log=log)
