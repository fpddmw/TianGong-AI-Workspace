"""Citation-centric helpers shared by the CLI and autonomous agents."""

from __future__ import annotations

import json
import math
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Optional, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from ..tooling.highly_cited import fetch_journal_citation_bands
from ..tooling.llm import ModelRouter
from ..tooling.mineru import MineruClient, MineruClientError
from ..tooling.openalex import LLMCitationAssessor, OpenAlexClient, OpenAlexClientError
from ..tooling.supabase import SupabaseClient, SupabaseClientError

__all__ = [
    "CitationStudyConfig",
    "JournalBandsConfig",
    "run_citation_study",
    "run_journal_bands_analysis",
    "load_works_file",
    "make_work_slug",
    "find_pdf_for_work",
    "extract_pdf_text",
    "stringify_supabase_result",
    "trim_text",
]


@dataclass(slots=True)
class CitationStudyConfig:
    works_file: Optional[Path] = None
    dois: Sequence[str] = ()
    pdf_dir: Optional[Path] = None
    mode: str = "supabase"
    supabase_top_k: int = 10
    supabase_est_k: int = 80
    temperature: float = 0.2
    pdf: Optional[Path] = None
    use_mineru: bool = False
    mineru_url: Optional[str] = None
    mineru_token: Optional[str] = None
    mineru_timeout: int = 300
    include_mineru_result: bool = False
    max_fulltext_chars: int = 30000


@dataclass(slots=True)
class JournalBandsConfig:
    journal_issn: str
    journal_name: str
    publish_year: Optional[int] = None
    band_limit: int = 200
    max_records: int = 800
    top_k: int = 10
    est_k: int = 80
    show_fulltext: bool = False
    output: Path = Path("journal_bands_summary.md")


def run_citation_study(config: CitationStudyConfig) -> Mapping[str, Any]:
    mode_normalized = config.mode.lower()
    if mode_normalized not in {"supabase", "pdf"}:
        raise ValueError("模式仅支持 'supabase' 或 'pdf'。")

    client = OpenAlexClient()
    router = ModelRouter()
    assessor = LLMCitationAssessor(router=router, temperature=config.temperature)

    supabase: Optional[SupabaseClient] = None
    if mode_normalized == "supabase":
        try:
            supabase = SupabaseClient()
        except SupabaseClientError as exc:
            raise SupabaseClientError(
                "Supabase 未配置，请更新 .sercrets/secrets.toml 或切换为 --mode pdf。"
            ) from exc

    works: list[Mapping[str, Any]] = []
    input_errors: list[str] = []
    input_source = "works-file"

    if config.works_file:
        works = load_works_file(config.works_file)
    elif config.dois:
        input_source = "doi"
        for value in config.dois:
            try:
                record = client.search_by_doi(value)
                works.append(record)
            except OpenAlexClientError as exc:
                input_errors.append(f"{value}: {exc}")
        if not works:
            error_detail = tuple(input_errors) if input_errors else None
            raise ValueError(f"未能从 OpenAlex 获取任何元数据，请检查 DOI 是否正确。{error_detail or ''}")
    else:
        raise ValueError("请提供 --works-file 或至少一个 --doi。")

    classified = []
    for work in works:
        figure_notes: Optional[str] = None
        raw_mineru_result: Optional[Mapping[str, Any]] = None
        pdf_path: Optional[Path] = None
        need_pdf = mode_normalized == "pdf" or config.use_mineru

        if need_pdf:
            if config.pdf and len(works) == 1:
                pdf_path = config.pdf
            elif isinstance(work, Mapping) and isinstance(work.get("pdf_path"), str):
                candidate = Path(work["pdf_path"])
                if candidate.exists():
                    pdf_path = candidate
            elif config.pdf_dir:
                pdf_path = find_pdf_for_work(work, config.pdf_dir)

            title_hint = work.get("title") or work.get("doi") or work.get("id") or "(unknown)"
            if mode_normalized == "pdf" and not pdf_path:
                raise ValueError(f"PDF 模式需要找到对应文件: {title_hint}")
            if config.use_mineru and mode_normalized == "supabase" and not pdf_path:
                raise ValueError(f"Supabase 模式启用 Mineru 时必须提供 PDF（--pdf 或 --pdf-dir）。缺少文件: {title_hint}")

        if config.use_mineru and pdf_path:
            try:
                mineru = MineruClient(
                    api_url_override=config.mineru_url,
                    token_override=config.mineru_token,
                    timeout=float(config.mineru_timeout),
                )
                mineru_prompt = (
                    "请逐条概括文档中的每个图表/配图，重点说明图表类型、呈现的数据或流程，以及它们如何帮助读者理解核心贡献。"
                    "不要臆造不存在的图表。输出简洁的要点列表。请使用英文回答。"
                )
                mineru_result = mineru.recognize_with_images(pdf_path, prompt=mineru_prompt)
                figure_payload = mineru_result.get("result")
                raw_mineru_result = mineru_result if config.include_mineru_result else None
                if isinstance(figure_payload, list):
                    figure_descriptions: list[str] = []
                    for entry in figure_payload:
                        if not isinstance(entry, Mapping):
                            continue
                        text = entry.get("text")
                        if not isinstance(text, str):
                            continue
                        if text.strip().lower().startswith("image description"):
                            page = entry.get("page_number")
                            prefix = f"[p{page}] " if page is not None else ""
                            figure_descriptions.append(prefix + text.strip())
                    if figure_descriptions:
                        figure_notes = "\n".join(figure_descriptions)
                    else:
                        figure_notes = json.dumps(figure_payload, ensure_ascii=False)
                else:
                    figure_notes = json.dumps(figure_payload, ensure_ascii=False)
            except MineruClientError as exc:
                figure_notes = None
                if config.use_mineru:
                    raise

        fulltext_used: Optional[str] = None
        fulltext_source = None
        fulltext_error: Optional[str] = None

        if mode_normalized == "supabase":
            doi_value = work.get("doi") or ""
            if not isinstance(doi_value, str) or not doi_value.strip():
                fulltext_error = "Supabase 模式需要 DOI。"
            elif supabase is None:
                fulltext_error = "Supabase 未初始化。"
            else:
                try:
                    supabase_result = supabase.fetch_content(
                        str(doi_value),
                        query="请尽可能均匀的在全文中选择十个段落",
                        top_k=config.supabase_top_k,
                        est_k=config.supabase_est_k,
                    )
                    fulltext_used = stringify_supabase_result(supabase_result, max_chars=config.max_fulltext_chars)
                    fulltext_source = "supabase"
                except SupabaseClientError as exc:
                    fulltext_error = str(exc)
        else:
            if pdf_path is None:
                fulltext_error = "PDF 模式缺少 PDF，无法提取全文。"
            else:
                try:
                    fulltext_used = extract_pdf_text(pdf_path, max_chars=config.max_fulltext_chars)
                    fulltext_source = "pdf"
                except OpenAlexClientError as exc:
                    fulltext_error = str(exc)

        llm_error: Optional[str] = None
        try:
            llm_result = assessor.assess(
                work,
                fulltext=fulltext_used,
                fulltext_source=fulltext_source or mode_normalized,
                figure_notes=trim_text(figure_notes, config.max_fulltext_chars // 2) if figure_notes else None,
            )
        except Exception as exc:
            llm_error = f"LLM 评分失败: {exc}"
            llm_result = {
                "prediction": None,
                "dimension_scores": None,
                "action_plan": None,
                "raw": None,
            }

        combined: MutableMapping[str, Any] = {
            "id": work.get("id"),
            "title": work.get("title"),
            "doi": work.get("doi"),
            "publication_year": work.get("publication_year"),
            "cited_by_count": work.get("cited_by_count"),
            "reference_count": work.get("referenced_works_count"),
            "prediction": llm_result.get("prediction"),
            "dimension_scores": llm_result.get("dimension_scores"),
            "action_plan": llm_result.get("action_plan"),
            "fulltext_mode": mode_normalized,
            "fulltext_source": fulltext_source or mode_normalized,
            "fulltext_excerpt_used": trim_text(fulltext_used, 800) if fulltext_used else None,
            "fulltext_error": fulltext_error,
            "figure_notes_used": trim_text(figure_notes, 800) if figure_notes else None,
            "pdf_path_used": str(pdf_path) if pdf_path else None,
            "mineru_result": raw_mineru_result if config.include_mineru_result else None,
            "llm_error": llm_error,
        }
        classified.append(combined)

    return {
        "query": None,
        "provider": "openalex",
        "input_source": input_source,
        "input_errors": input_errors or None,
        "count": len(classified),
        "works": classified,
    }


def run_journal_bands_analysis(config: JournalBandsConfig, *, log: Callable[[str], None] | None = None) -> Path:
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

    supabase = SupabaseClient()
    router = ModelRouter()
    chat_model = router.create_chat_model(purpose="general", temperature=0.2)

    feature_eval_schema: Mapping[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "eval": {"type": "string", "enum": ["优", "中", "差"]},
            "reason": {"type": "string"},
        },
        "required": ["eval", "reason"],
    }
    response_schema: Mapping[str, Any] = {
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
    header_lines.append(f"- Band limit: {config.band_limit} (after percentile split), max records considered: {config.max_records}")
    header_lines.append(
        f"- Citation thresholds: p25={bands.thresholds.get('p25', 0):.1f}, p75={bands.thresholds.get('p75', 0):.1f}"
    )
    header_lines.append(f"- Supabase sci_search: topK={config.top_k}, estK={config.est_k}")
    header_lines.append("")
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(header_lines) + "\n")

    def _sample_band(items: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        if not items:
            return []
        sample_size = max(1, int(math.ceil(len(items) * 0.1)))
        return random.sample(items, min(sample_size, len(items)))

    for section_title, works in sections:
        section_lines: list[str] = [f"## {section_title}"]
        if not works:
            section_lines.append("_No records found._")
            section_lines.append("")
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(section_lines))
            continue
        sampled_works = _sample_band(works)
        section_lines.append(f"_Sampled {len(sampled_works)} of {len(works)} papers (10% random)._")
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(section_lines) + "\n")

        _log(f"Processing {len(sampled_works)} sampled papers (source pool: {len(works)}) in {section_title}...")

        def _analyze_work(task: tuple[int, Mapping[str, Any]]) -> tuple[int, list[str]]:
            idx, work = task
            doi = work.get("doi")
            title = work.get("title") or "(untitled)"
            citation_count = work.get("cited_by_count")
            year = work.get("publication_year")
            try:
                fulltext = supabase.fetch_content(
                    str(doi or ""),
                    query="请尽可能均匀的在全文中选择十个段落",
                    top_k=config.top_k,
                    est_k=config.est_k,
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
                resp = chat_model.invoke(
                    messages, response_format={"type": "json_schema", "json_schema": response_schema}
                )
                content = getattr(resp, "content", None) or (
                    resp.generations[0].message.content if hasattr(resp, "generations") else None
                )
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
            except Exception as exc:
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

        tasks = list(enumerate(sampled_works, start=1))
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(_analyze_work, tasks))

        for idx, entry_lines in sorted(results, key=lambda item: item[0]):
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(entry_lines))
            if idx % 5 == 0 or idx == len(sampled_works):
                _log(f"{section_title}: {idx}/{len(sampled_works)} written")

    _log(f"Summary streaming written to {output_path}")
    return output_path


def load_works_file(path: Path) -> list[Mapping[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in works file: {exc}") from exc
    works = None
    if isinstance(data, Mapping):
        if "works" in data:
            works = data.get("works")
        elif "payload" in data and isinstance(data.get("payload"), Mapping) and "works" in data["payload"]:
            works = data["payload"]["works"]
        else:
            works = data
    else:
        works = data
    if not isinstance(works, list):
        raise ValueError("Works file must contain a JSON array or an object with a 'works' array.")
    normalized: list[Mapping[str, Any]] = []
    for item in works:
        if isinstance(item, Mapping):
            normalized.append(item)
        else:
            raise ValueError("Each work entry must be a JSON object.")
    return normalized


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return cleaned or "work"


def make_work_slug(work: Mapping[str, Any], fallback: str) -> str:
    doi = work.get("doi")
    if isinstance(doi, str) and doi:
        return _slugify(doi.replace("/", "_"))
    work_id = work.get("id")
    if isinstance(work_id, str) and work_id:
        return _slugify(work_id.rsplit("/", 1)[-1])
    title = work.get("title")
    if isinstance(title, str) and title:
        return _slugify(title)
    return _slugify(fallback)


def trim_text(value: Optional[str], max_chars: int) -> Optional[str]:
    if value is None:
        return None
    text = value.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[truncated {len(text) - max_chars} chars]"


def stringify_supabase_result(result: Mapping[str, Any], max_chars: int) -> str:
    if "data" in result and isinstance(result.get("data"), list):
        parts: list[str] = []
        for item in result["data"]:
            if not isinstance(item, Mapping):
                continue
            fragment = item.get("content") or item.get("text") or item.get("chunk")
            if fragment:
                parts.append(str(fragment))
        if parts:
            combined = "\n\n".join(parts)
            trimmed = trim_text(combined, max_chars)
            if trimmed:
                return trimmed
    fallback = json.dumps(result, ensure_ascii=False)
    return trim_text(fallback, max_chars) or ""


def extract_pdf_text(pdf_path: Path, max_chars: int = 12000) -> str:
    if not pdf_path.exists():
        raise OpenAlexClientError(f"PDF not found: {pdf_path}")
    try:
        from pypdf import PdfReader
    except Exception as exc:  # pragma: no cover - defensive
        raise OpenAlexClientError(f"Failed to read PDF {pdf_path}: {exc}") from exc
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:  # pragma: no cover - defensive
        raise OpenAlexClientError(f"Failed to read PDF {pdf_path}: {exc}") from exc

    snippets: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            continue
        cleaned = text.strip()
        if cleaned:
            snippets.append(cleaned)
        if len(" ".join(snippets)) >= max_chars:
            break
    combined = "\n".join(snippets).strip()
    if not combined:
        raise OpenAlexClientError(f"未能从 PDF 提取文本: {pdf_path}")
    return trim_text(combined, max_chars) or ""


def find_pdf_for_work(work: Mapping[str, Any], pdf_dir: Path) -> Optional[Path]:
    """Match a PDF in pdf_dir by DOI, OpenAlex ID, or title."""

    def _candidate_slugs(value: str) -> list[str]:
        variants = {value}
        variants.add(value.replace("/", "_"))
        variants.add(value.replace("/", "-"))
        variants.add(value.replace(".", "-"))
        if "/" in value:
            variants.add(value.rsplit("/", 1)[-1])
        return [_slugify(v) for v in variants if v]

    candidates: list[str] = []
    doi = work.get("doi")
    if isinstance(doi, str):
        candidates.extend(_candidate_slugs(doi))
    work_id = work.get("id")
    if isinstance(work_id, str):
        candidates.extend(_candidate_slugs(work_id))
    title = work.get("title")
    if isinstance(title, str):
        candidates.extend(_candidate_slugs(title))
    pdf_url = work.get("pdf_url")
    if isinstance(pdf_url, str):
        candidates.extend(_candidate_slugs(pdf_url))

    normalized_candidates = [c for c in candidates if c]
    if not normalized_candidates:
        return None

    pdf_files = list(pdf_dir.glob("*.pdf")) + list(pdf_dir.glob("*.PDF"))
    for pdf_path in pdf_files:
        name_key = _slugify(pdf_path.stem)
        full_key = _slugify(pdf_path.name)
        if any(key and (key in name_key or key in full_key) for key in normalized_candidates):
            return pdf_path
    return None
