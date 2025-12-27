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
    "doi_to_filename",
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


def doi_to_filename(doi: str, *, suffix: str = ".md") -> str:
    value = doi.strip()
    value = re.sub(r"^https?://(dx\.)?doi\.org/", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^doi:\\s*", "", value, flags=re.IGNORECASE)
    value = value.strip().lower().replace("/", "_")
    value = re.sub(r"[^a-z0-9._-]+", "_", value).strip("._-")
    if not value:
        value = "work"
    if suffix and not suffix.startswith("."):
        suffix = "." + suffix
    return f"{value}{suffix or ''}"


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
            "score": {"type": "integer", "minimum": 1, "maximum": 5},
            "analysis": {"type": "string"},
        },
        "required": ["score", "analysis"],
    }


def _impact_schema() -> Mapping[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "score": {"type": "integer", "minimum": 1, "maximum": 5},
            "level": {"type": "string", "enum": ["低", "中", "高", "极高"]},
            "lift_factors": {"type": "array", "items": {"type": "string"}, "minItems": 2},
            "drag_factors": {"type": "array", "items": {"type": "string"}, "minItems": 2},
            "summary": {"type": "string"},
        },
        "required": ["score", "level", "lift_factors", "drag_factors", "summary"],
    }


def _render_report(parsed: Mapping[str, Any], *, title: str) -> str:
    template = CITATION_TEMPLATE_PATH.read_text(encoding="utf-8")

    def _fmt_markdown_list(values: Any) -> str:
        if isinstance(values, str):
            values = [values]
        if not values:
            return "- 无"
        items = [str(v).strip() for v in values if str(v).strip()]
        return "\n".join(f"- {v}" for v in items) if items else "- 无"

    def _fmt_inline_list(values: Any) -> str:
        if isinstance(values, str):
            values = [values]
        if not values:
            return "无"
        items = [str(v).strip() for v in values if str(v).strip()]
        return "、".join(items) if items else "无"

    def _fmt_dimension_block() -> str:
        dimension_labels = [
            ("c1_pain_point", "C1 痛点命中度"),
            ("c2_actionability", "C2 可操作性"),
            ("c3_reusability", "C3 可复用资产"),
            ("c4_citability", "C4 可引用性/可教学性"),
            ("c5_evidence_strength", "C5 证据强度"),
            ("c6_generalizability", "C6 泛化与外推能力"),
            ("c7_uncertainty_auditability", "C7 不确定性表达与可审计性"),
            ("c8_diffusion", "C8 跨圈层扩散潜力"),
            ("c9_standardization", "C9 标准化/制度化进入性"),
            ("c10_ecosystem", "C10 持续生态潜力"),
        ]
        dims = parsed.get("dimensions") if isinstance(parsed.get("dimensions"), Mapping) else {}
        lines: list[str] = []
        for key, label in dimension_labels:
            dim = dims.get(key, {}) if isinstance(dims, Mapping) else {}
            score = dim.get("score")
            analysis = dim.get("analysis") or ""
            score_text = f"{score}/5" if isinstance(score, int) else "n/a"
            lines.append(f"- {label}: {score_text} — {analysis}".strip(" —"))
        return "\n".join(lines) if lines else "- 无"

    def _fmt_impact(value: Any) -> Mapping[str, str]:
        data = value if isinstance(value, Mapping) else {}
        score = data.get("score")
        score_text = f"{score}/5" if isinstance(score, int) else "n/a"
        return {
            "score": score_text,
            "level": data.get("level") or "n/a",
            "lifts": _fmt_markdown_list(data.get("lift_factors")),
            "drags": _fmt_markdown_list(data.get("drag_factors")),
            "summary": data.get("summary") or "未提供总结。",
        }

    early = _fmt_impact(parsed.get("early_impact"))
    five_year = _fmt_impact(parsed.get("five_year_impact"))

    replacements = {
        "title": title,
        "research_types": _fmt_inline_list(parsed.get("research_types")),
        "secondary_types": _fmt_inline_list(parsed.get("secondary_types")),
        "dimensions_block": _fmt_dimension_block(),
        "early_score": early["score"],
        "early_level": early["level"],
        "early_lifts": early["lifts"],
        "early_drags": early["drags"],
        "early_summary": early["summary"],
        "five_score": five_year["score"],
        "five_level": five_year["level"],
        "five_lifts": five_year["lifts"],
        "five_drags": five_year["drags"],
        "five_summary": five_year["summary"],
        "impact_pathways": _fmt_markdown_list(parsed.get("impact_pathways")),
        "risks": _fmt_markdown_list(parsed.get("risks")),
        "recommendations": _fmt_markdown_list(parsed.get("recommendations")),
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
                "research_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["机制/发现型", "方法/模型型", "数据/基础设施型", "治理/框架/话语型"],
                    },
                    "minItems": 1,
                },
                "secondary_types": {"type": "array", "items": {"type": "string"}},
                "dimensions": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "c1_pain_point": _dimension_schema(),
                        "c2_actionability": _dimension_schema(),
                        "c3_reusability": _dimension_schema(),
                        "c4_citability": _dimension_schema(),
                        "c5_evidence_strength": _dimension_schema(),
                        "c6_generalizability": _dimension_schema(),
                        "c7_uncertainty_auditability": _dimension_schema(),
                        "c8_diffusion": _dimension_schema(),
                        "c9_standardization": _dimension_schema(),
                        "c10_ecosystem": _dimension_schema(),
                    },
                    "required": [
                        "c1_pain_point",
                        "c2_actionability",
                        "c3_reusability",
                        "c4_citability",
                        "c5_evidence_strength",
                        "c6_generalizability",
                        "c7_uncertainty_auditability",
                        "c8_diffusion",
                        "c9_standardization",
                        "c10_ecosystem",
                    ],
                },
                "early_impact": _impact_schema(),
                "five_year_impact": _impact_schema(),
                "impact_pathways": {"type": "array", "items": {"type": "string"}},
                "risks": {"type": "array", "items": {"type": "string"}},
                "recommendations": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "research_types",
                "secondary_types",
                "dimensions",
                "early_impact",
                "five_year_impact",
                "impact_pathways",
                "risks",
                "recommendations",
            ],
        },
        "strict": True,
    }

    system_prompt = (
        "你是一名严格、客观、但不回避给出高低判断的研究评审模型。 "
        "你的目标不是“谨慎”，而是做出有区分度、可解释、可复核的影响力判断。 "
        "禁止行为： - 默认给“中” - 因缺乏引用或工具而一刀切降级 - 给出“全是优点、没有限制”的结论"
    )
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
