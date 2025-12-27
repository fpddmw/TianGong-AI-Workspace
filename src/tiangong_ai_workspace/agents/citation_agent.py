"""Citation-centric helpers shared by the CLI and autonomous agents."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Optional, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from ..tooling.llm import ModelRouter
from ..tooling.mineru import MineruClient, MineruClientError
from ..tooling.openalex import LLMCitationAssessor, OpenAlexClient, OpenAlexClientError
from ..tooling.get_fulltext import SupabaseClient, SupabaseClientError
from .journal_bands_agent import JournalBandsAgent, JournalBandsConfig, run_journal_bands_agent

__all__ = [
    "CitationStudyConfig",
    "JournalBandsConfig",
    "JournalBandsAgent",
    "CitationTextReportConfig",
    "run_citation_study",
    "run_journal_bands_analysis",
    "run_journal_bands_agent",
    "generate_citation_text_report",
    "load_works_file",
    "make_work_slug",
    "find_pdf_for_work",
    "extract_pdf_text",
    "stringify_supabase_result",
    "trim_text",
]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCORE_CRITERIA_PATH = PROJECT_ROOT / "prompts" / "citation_prediction" / "score_criteria.md"
CITATION_TEMPLATE_PATH = PACKAGE_ROOT / "templates" / "citation_impact_report.md"
MAX_REPORT_TEXT_CHARS = 50000


@dataclass(slots=True)
class CitationStudyConfig:
    works_file: Optional[Path] = None
    dois: Sequence[str] = ()
    pdf_dir: Optional[Path] = None
    mode: str = "supabase"
    supabase_top_k: int = 10
    supabase_ext_k: int = 80
    temperature: float = 0.2
    pdf: Optional[Path] = None
    use_mineru: bool = False
    mineru_url: Optional[str] = None
    mineru_token: Optional[str] = None
    mineru_timeout: int = 300
    include_mineru_result: bool = False
    max_fulltext_chars: int = 30000


@dataclass(slots=True)
class CitationTextReportConfig:
    """Plain-text citation impact report configuration."""

    text_path: Path
    title: Optional[str] = None
    temperature: float = 0.3

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
                        top_k=config.supabase_top_k,
                        ext_k=config.supabase_ext_k,
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
    """Backwards-compatible wrapper around the dedicated journal bands agent."""

    return run_journal_bands_agent(config, log=log)


def generate_citation_text_report(config: CitationTextReportConfig) -> Mapping[str, Any]:
    if not config.text_path.exists():
        raise ValueError(f"Text file not found: {config.text_path}")

    try:
        paper_text_raw = config.text_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Failed to read text file {config.text_path}: {exc}") from exc

    paper_text = trim_text(paper_text_raw, MAX_REPORT_TEXT_CHARS) or ""
    if not paper_text.strip():
        raise ValueError("Paper text is empty.")

    criteria_text = _load_score_criteria()
    router = ModelRouter()
    chat_model = router.create_chat_model(purpose="general", temperature=config.temperature)

    schema: Mapping[str, Any] = {
        "name": "citation_impact_report",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "research_types": {"type": "array", "items": {"type": "string"}},
                "secondary_types": {"type": "array", "items": {"type": "string"}},
                "dimensions": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "topic_frontier": _dimension_schema(),
                        "methodology": _dimension_schema(),
                        "data_evidence": _dimension_schema(),
                        "conclusion_depth": _dimension_schema(),
                        "presentation": _dimension_schema(),
                    },
                    "required": ["topic_frontier", "methodology", "data_evidence", "conclusion_depth", "presentation"],
                },
                "early_impact": _impact_schema(),
                "five_year_impact": _impact_schema(),
                "impact_pathways": {"type": "array", "items": {"type": "string"}},
                "risks": {"type": "array", "items": {"type": "string"}},
                "recommendations": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["research_types", "dimensions", "early_impact", "five_year_impact"],
        },
        "strict": True,
    }

    system_prompt = (
        "你是一名可持续发展与环境领域的资深研究评估专家。"
        "请根据提供的评分规范，对给定论文文本输出结构化 JSON，随后将用于渲染报告。"
        "保持客观、具体，引用文本中的关键信息。"
    )
    user_prompt = (
        f"评分规范:\n{criteria_text}\n\n"
        f"论文全文（纯文本，图表已转为文字，可按需引用）：\n{paper_text}"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    raw = chat_model.invoke(messages, response_format={"type": "json_schema", "json_schema": schema})
    content = getattr(raw, "content", None) or (raw.generations[0].message.content if hasattr(raw, "generations") else None)
    parsed: Mapping[str, Any]
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse model response as JSON: {exc}") from exc
    elif isinstance(content, Mapping):
        parsed = dict(content)
    else:
        raise ValueError("Unexpected response format from model.")

    report_text = _render_report(parsed, title=config.title or config.text_path.stem)
    return {
        "title": config.title or config.text_path.stem,
        "structured": parsed,
        "report": report_text,
    }


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


def _load_score_criteria() -> str:
    if not SCORE_CRITERIA_PATH.exists():
        raise FileNotFoundError(f"Score criteria file not found: {SCORE_CRITERIA_PATH}")
    return SCORE_CRITERIA_PATH.read_text(encoding="utf-8")


def _dimension_schema() -> Mapping[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "score": {"type": "integer", "minimum": 0, "maximum": 3},
            "analysis": {"type": "string"},
        },
        "required": ["score", "analysis"],
    }


def _impact_schema() -> Mapping[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "level": {"type": "string", "enum": ["低", "中", "高"]},
            "analysis": {"type": "string"},
        },
        "required": ["level", "analysis"],
    }


def _render_report(parsed: Mapping[str, Any], *, title: str) -> str:
    template = CITATION_TEMPLATE_PATH.read_text(encoding="utf-8")

    def _fmt_dimension(key: str) -> str:
        dim = parsed.get("dimensions", {}).get(key, {}) if isinstance(parsed.get("dimensions"), Mapping) else {}
        score = dim.get("score")
        analysis = dim.get("analysis") or ""
        score_text = f"{score}/3" if isinstance(score, int) else "n/a"
        return f"{score_text} — {analysis}".strip(" —")

    def _fmt_list(values: Any) -> str:
        if isinstance(values, str):
            values = [values]
        if not values:
            return "- 无"
        items = [str(v).strip() for v in values if str(v).strip()]
        return "\n".join(f"- {v}" for v in items) if items else "- 无"

    early = parsed.get("early_impact", {}) if isinstance(parsed.get("early_impact"), Mapping) else {}
    five_year = parsed.get("five_year_impact", {}) if isinstance(parsed.get("five_year_impact"), Mapping) else {}

    replacements = {
        "title": title,
        "research_types": _fmt_list(parsed.get("research_types")),
        "secondary_types": _fmt_list(parsed.get("secondary_types")),
        "topic_frontier": _fmt_dimension("topic_frontier"),
        "methodology": _fmt_dimension("methodology"),
        "data_evidence": _fmt_dimension("data_evidence"),
        "conclusion_depth": _fmt_dimension("conclusion_depth"),
        "presentation": _fmt_dimension("presentation"),
        "early_level": early.get("level") or "n/a",
        "early_analysis": early.get("analysis") or "未提供分析。",
        "five_level": five_year.get("level") or "n/a",
        "five_analysis": five_year.get("analysis") or "未提供分析。",
        "impact_pathways": _fmt_list(parsed.get("impact_pathways")),
        "risks": _fmt_list(parsed.get("risks")),
        "recommendations": _fmt_list(parsed.get("recommendations")),
    }

    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(f"{{{{ {key} }}}}", str(value))
    return rendered
