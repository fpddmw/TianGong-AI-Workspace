"""Citation-centric helpers shared by the CLI and autonomous agents."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from ..tooling.get_fulltext import SupabaseClient, SupabaseClientError
from ..tooling.llm import ModelRouter
from ..tooling.mineru import MineruClient

__all__ = [
    "CitationStudyConfig",
    "FullTextResult",
    "CitationTextReportConfig",
    "run_citation_study",
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
class FullTextResult:
    """Normalized fulltext payload used by scoring helpers."""

    text: str
    mode: str
    source: str
    raw: Optional[Mapping[str, Any]] = None


@dataclass(slots=True)
class CitationTextReportConfig:
    """Plain-text citation impact report configuration."""

    text_path: Path
    title: Optional[str] = None
    temperature: float = 0.3


def run_citation_study(config: CitationStudyConfig) -> Mapping[str, Any]:
    """Fetch fulltext (Supabase by DOI or PDF via Mineru) and score it with the rubric."""

    works = _resolve_works(config)
    results: list[Mapping[str, Any]] = []
    for idx, work in enumerate(works):
        title = _resolve_title(work, idx)
        fulltext = _fetch_fulltext_for_work(work, config)
        scored = _score_text(
            fulltext.text,
            title=title,
            temperature=config.temperature,
        )
        entry: MutableMapping[str, Any] = {
            "title": title,
            "doi": work.get("doi"),
            "fulltext_mode": fulltext.mode,
            "fulltext_source": fulltext.source,
            "fulltext_excerpt": trim_text(fulltext.text, config.max_fulltext_chars),
            "report": scored["report"],
            "structured": scored["structured"],
        }
        if fulltext.raw is not None:
            entry["raw_fulltext"] = fulltext.raw
        results.append(entry)

    return {
        "works": results,
        "count": len(results),
    }


def generate_citation_text_report(config: CitationTextReportConfig) -> Mapping[str, Any]:
    if not config.text_path.exists():
        raise ValueError(f"Text file not found: {config.text_path}")

    try:
        paper_text_raw = config.text_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Failed to read text file {config.text_path}: {exc}") from exc

    return _score_text(
        paper_text_raw,
        title=config.title or config.text_path.stem,
        temperature=config.temperature,
    )


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
        raise ValueError(f"PDF not found: {pdf_path}")
    try:
        from pypdf import PdfReader
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Failed to read PDF {pdf_path}: {exc}") from exc
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Failed to read PDF {pdf_path}: {exc}") from exc

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
        raise ValueError(f"未能从 PDF 提取文本: {pdf_path}")
    return trim_text(combined, max_chars) or ""


def find_pdf_for_work(work: Mapping[str, Any], pdf_dir: Path) -> Optional[Path]:
    """Match a PDF in pdf_dir by DOI, work ID, or title."""

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


def _resolve_works(config: CitationStudyConfig) -> list[Mapping[str, Any]]:
    if config.works_file:
        return load_works_file(config.works_file)
    if config.dois:
        return [{"doi": doi} for doi in config.dois]
    if config.pdf:
        return [{"pdf_path": config.pdf}]
    pdf_dir = config.pdf_dir or Path("input")
    if pdf_dir.exists() and pdf_dir.is_dir():
        pdfs = sorted(list(pdf_dir.glob("*.pdf")) + list(pdf_dir.glob("*.PDF")))
        return [{"pdf_path": pdf} for pdf in pdfs]
    raise ValueError("No works provided. Specify --doi, --works-file, --pdf, or ensure a PDF exists in ./input.")


def _resolve_title(work: Mapping[str, Any], idx: int) -> str:
    title = work.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    doi = work.get("doi")
    if isinstance(doi, str) and doi.strip():
        return doi.strip()
    pdf_path = work.get("pdf_path")
    if isinstance(pdf_path, (str, Path)):
        return Path(pdf_path).stem
    return f"Work {idx + 1}"


def _fetch_fulltext_for_work(work: Mapping[str, Any], config: CitationStudyConfig) -> FullTextResult:
    doi = work.get("doi")
    pdf_path = _resolve_pdf_path(work, config)

    if config.mode == "pdf":
        if pdf_path:
            return _fetch_pdf_fulltext(pdf_path, config)
        if isinstance(doi, str) and doi.strip():
            return _fetch_supabase_fulltext(doi.strip(), config)
    else:
        if isinstance(doi, str) and doi.strip():
            return _fetch_supabase_fulltext(doi.strip(), config)
        if pdf_path:
            return _fetch_pdf_fulltext(pdf_path, config)

    raise ValueError("Work is missing DOI and PDF; unable to fetch fulltext.")


def _resolve_pdf_path(work: Mapping[str, Any], config: CitationStudyConfig) -> Optional[Path]:
    pdf_path = work.get("pdf_path")
    if isinstance(pdf_path, (str, Path)):
        path = Path(pdf_path)
        if path.exists():
            return path
    pdf_dir = config.pdf_dir or Path("input")
    if pdf_dir.exists():
        matched = find_pdf_for_work(work, pdf_dir)
        if matched:
            return matched
    if config.pdf and config.pdf.exists():
        return config.pdf
    return None


def _fetch_supabase_fulltext(doi: str, config: CitationStudyConfig) -> FullTextResult:
    supabase = SupabaseClient()
    result = supabase.fetch_content(doi, top_k=config.supabase_top_k, ext_k=config.supabase_ext_k)
    text = stringify_supabase_result(result, config.max_fulltext_chars)
    if not text:
        raise SupabaseClientError("Supabase returned no text.")
    raw_payload = result if config.include_mineru_result else None
    return FullTextResult(text=text, mode="supabase", source=doi, raw=raw_payload)


def _fetch_pdf_fulltext(pdf_path: Path, config: CitationStudyConfig) -> FullTextResult:
    if config.use_mineru:
        mineru = MineruClient(
            api_url_override=config.mineru_url,
            token_override=config.mineru_token,
            timeout=float(config.mineru_timeout),
        )
        mineru_result = mineru.recognize_with_images(pdf_path)
        text = _extract_mineru_text(mineru_result, config.max_fulltext_chars)
        if not text:
            text = extract_pdf_text(pdf_path, config.max_fulltext_chars)
        raw_payload = mineru_result if config.include_mineru_result else None
        return FullTextResult(text=text, mode="mineru", source=str(pdf_path), raw=raw_payload)

    text = extract_pdf_text(pdf_path, config.max_fulltext_chars)
    return FullTextResult(text=text, mode="pdf", source=str(pdf_path))


def _extract_mineru_text(payload: Mapping[str, Any], max_chars: int) -> str:
    if not isinstance(payload, Mapping):
        return ""
    result = payload.get("result")
    candidates: list[str] = []
    if isinstance(result, Mapping):
        for key in ("md_txt", "markdown", "text", "txt", "content"):
            value = result.get(key)
            if isinstance(value, str):
                candidates.append(value)
            elif isinstance(value, list):
                joined = "\n".join(str(item) for item in value if str(item).strip())
                if joined:
                    candidates.append(joined)
    if not candidates and result is not None:
        candidates.append(json.dumps(result, ensure_ascii=False))
    if not candidates:
        return ""
    combined = "\n\n".join(candidates)
    return trim_text(combined, max_chars) or ""


def _score_text(paper_text_raw: str, *, title: str, temperature: float) -> Mapping[str, Any]:
    paper_text = trim_text(paper_text_raw, MAX_REPORT_TEXT_CHARS) or ""
    if not paper_text.strip():
        raise ValueError("Paper text is empty.")

    criteria_text = _load_score_criteria()
    router = ModelRouter()
    chat_model = router.create_chat_model(purpose="general", temperature=temperature)

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

    system_prompt = "你是一名可持续发展与环境领域的资深研究评估专家。" "请根据提供的评分规范，对给定论文文本输出结构化 JSON，随后将用于渲染报告。" "保持客观、具体，引用文本中的关键信息。"
    user_prompt = f"评分规范:\n{criteria_text}\n\n" f"论文全文（纯文本，图表已转为文字，可按需引用）：\n{paper_text}"

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

    report_text = _render_report(parsed, title=title)
    return {
        "title": title,
        "structured": parsed,
        "report": report_text,
    }
