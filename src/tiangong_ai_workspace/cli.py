"""
Command line utilities for the Tiangong AI Workspace.

The CLI provides quick checks for local prerequisites (Python, uv, Node.js)
and lists the external AI tooling CLIs that this workspace integrates with.
Edit this file to tailor the workspace to your own toolchain.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence

import typer
from langchain_core.messages import HumanMessage

from . import __version__
from .agents import DocumentWorkflowConfig, DocumentWorkflowType, run_document_workflow
from .agents.citation_agent import (
    CitationStudyConfig,
    CitationTextReportConfig,
    generate_citation_text_report,
    make_work_slug,
    run_citation_study,
)
from .agents.deep_agent import build_workspace_deep_agent
from .agents.journal_bands_agent import JournalBandsConfig, run_journal_bands_agent
from .mcp_client import MCPToolClient
from .secrets import MCPServerSecrets, discover_secrets_path, load_secrets
from .tooling import WorkspaceResponse, list_registered_tools
from .tooling.config import CLIToolConfig, load_workspace_config
from .tooling.crossref import CrossrefClient, CrossrefClientError
from .tooling.dify import DifyKnowledgeBaseClient, DifyKnowledgeBaseError
from .tooling.embeddings import OpenAICompatibleEmbeddingClient, OpenAIEmbeddingError
from .tooling.gemini import GeminiDeepResearchClient, GeminiDeepResearchError
from .tooling.get_fulltext import SupabaseClientError
from .tooling.llm import ModelPurpose
from .tooling.mineru import MineruClient, MineruClientError
from .tooling.openalex import OpenAlexClient, OpenAlexClientError
from .tooling.tavily import TavilySearchClient, TavilySearchError

app = typer.Typer(help="Tiangong AI Workspace CLI for managing local AI tooling.")
mcp_app = typer.Typer(help="Interact with Model Context Protocol services configured for this workspace.")
app.add_typer(mcp_app, name="mcp")
docs_app = typer.Typer(help="Document-generation workflows driven by LangChain/LangGraph.")
app.add_typer(docs_app, name="docs")
agents_app = typer.Typer(help="General-purpose workspace agent workflows.")
app.add_typer(agents_app, name="agents")
knowledge_app = typer.Typer(help="Knowledge base utilities such as Dify dataset retrieval.")
app.add_typer(knowledge_app, name="knowledge")
embeddings_app = typer.Typer(help="OpenAI-compatible embedding helpers.")
app.add_typer(embeddings_app, name="embeddings")
crossref_app = typer.Typer(help="Crossref metadata utilities.")
app.add_typer(crossref_app, name="crossref")
openalex_app = typer.Typer(help="OpenAlex metadata utilities.")
app.add_typer(openalex_app, name="openalex")
gemini_app = typer.Typer(help="Gemini API helpers including the Deep Research agent.")
app.add_typer(gemini_app, name="gemini")

WORKFLOW_SUMMARIES = {
    DocumentWorkflowType.REPORT: "Business and technical reports with clear recommendations.",
    DocumentWorkflowType.PATENT_DISCLOSURE: "Patent disclosure drafts capturing inventive details.",
    DocumentWorkflowType.PLAN: "Execution or project plans with milestones and risks.",
    DocumentWorkflowType.PROJECT_PROPOSAL: "Project proposals optimised for stakeholder buy-in.",
}


def _get_version(command: str, version_args: Sequence[str] | None = None) -> str | None:
    """
    Return the version string for a CLI command if available.

    Many CLIs support `--version` and emit to stdout; others may use stderr.
    """
    try:
        args = [command]
        args.extend(version_args or ("--version",))
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None

    output = (result.stdout or result.stderr).strip()
    return output or None


@app.command()
def info() -> None:
    """Print a short summary of the workspace."""
    typer.echo(f"Tiangong AI Workspace v{__version__}")
    typer.echo("Unified CLI workspace for Codex, Gemini, and Claude Code automation.")
    typer.echo("")
    typer.echo(f"Project root : {Path.cwd()}")
    typer.echo(f"Python       : {sys.version.split()[0]} (requires >=3.12)")
    uv_path = shutil.which("uv")
    typer.echo(f"uv executable: {uv_path or 'not found in PATH'}")


@app.command("tools")
def list_tools(
    catalog: bool = typer.Option(
        False,
        "--catalog",
        help="Show the internal agent tool registry instead of local CLI commands.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit a machine-readable JSON response."),
) -> None:
    """List the external AI tooling CLIs tracked by the workspace or the agent catalog."""

    if catalog:
        registry = list_registered_tools()
        items = [
            {
                "name": descriptor.name,
                "description": descriptor.description,
                "category": descriptor.category,
                "entrypoint": descriptor.entrypoint,
                "tags": list(descriptor.tags),
            }
            for descriptor in registry.values()
        ]
        response = WorkspaceResponse.ok(
            payload={"tools": items},
            message="Workspace agent tool registry.",
            source="catalog",
        )
        _emit_response(response, json_output)
        if not json_output:
            typer.echo("")
            for item in items:
                typer.echo(f"- {item['name']}: {item['description']} [{item['category']}]")
        return

    cli_tools = []
    for tool in _cli_tool_configs():
        location = shutil.which(tool.command)
        version = _get_version(tool.command, tool.version_args) if location else None
        cli_tools.append(
            {
                "command": tool.command,
                "label": tool.label,
                "installed": bool(location),
                "location": location,
                "version": version,
            }
        )

    if json_output:
        response = WorkspaceResponse.ok(payload={"cli_tools": cli_tools}, message="CLI tooling status.", source="local")
        _emit_response(response, json_output=True)
        return

    typer.echo("Configured AI tooling commands:")
    for info in cli_tools:
        status = "[OK]" if info["installed"] else "[MISSING]"
        detail = info["version"] or "not installed"
        typer.echo(f"- {info['label']}: `{info['command']}` {status} ({detail})")
    typer.echo("")
    typer.echo("Edit [tool.tiangong.workspace.cli_tools] in pyproject.toml to customize this list.")


@app.command()
def check() -> None:
    """Validate local prerequisites such as Python, uv, Node.js, and AI CLIs."""
    typer.echo("Checking workspace prerequisites...\n")

    python_ok = sys.version_info >= (3, 12)
    python_status = "[OK]" if python_ok else "[WARN]"
    typer.echo(f"{python_status} Python {sys.version.split()[0]} (requires >=3.12)")

    uv_path = shutil.which("uv")
    uv_status = "[OK]" if uv_path else "[MISSING]"
    typer.echo(f"{uv_status} Astral uv: {uv_path or 'not found'}")

    node_path = shutil.which("node")
    if node_path:
        node_version = _get_version("node") or "version unknown"
        typer.echo(f"[OK] Node.js: {node_version} ({node_path})")
    else:
        typer.echo("[MISSING] Node.js: required for Node-based CLIs such as Claude Code")

    typer.echo("")
    typer.echo("AI coding toolchains:")
    for tool in _cli_tool_configs():
        location = shutil.which(tool.command)
        status = "[OK]" if location else "[MISSING]"
        version = _get_version(tool.command, tool.version_args) if location else None
        detail = version or "not installed"
        typer.echo(f"{status} {tool.label} ({tool.command}): {location or detail}")

    typer.echo("")
    typer.echo("Update [tool.tiangong.workspace.cli_tools] in pyproject.toml to adjust tool detection rules.")


@docs_app.command("list")
def docs_list(
    json_output: bool = typer.Option(False, "--json", help="Emit a machine-readable JSON response."),
) -> None:
    """List supported document-generation workflows."""

    items = []
    for workflow in DocumentWorkflowType:
        items.append(
            {
                "value": workflow.value,
                "tone": workflow.prompt_tone,
                "description": WORKFLOW_SUMMARIES.get(workflow, ""),
                "template": workflow.template_name,
            }
        )

    response = WorkspaceResponse.ok(payload={"workflows": items}, message="Available document workflows.")
    _emit_response(response, json_output)

    if not json_output:
        typer.echo("")
        typer.echo("Run `uv run tiangong-workspace docs run --help` to generate a document.")


@agents_app.command("list")
def agents_list(
    json_output: bool = typer.Option(False, "--json", help="Emit a machine-readable JSON response."),
) -> None:
    """List workspace-level agents and runtime executors."""

    registry = list_registered_tools()
    items = [
        {
            "name": descriptor.name,
            "description": descriptor.description,
            "category": descriptor.category,
            "entrypoint": descriptor.entrypoint,
            "tags": list(descriptor.tags),
        }
        for descriptor in registry.values()
        if descriptor.category in {"agent", "runtime"}
    ]

    response = WorkspaceResponse.ok(payload={"agents": items}, message="Available agents and runtime executors.")
    _emit_response(response, json_output)

    if not json_output:
        typer.echo("")
        for item in items:
            typer.echo(f"- {item['name']}: {item['description']} [{item['category']}]")


@agents_app.command("run")
def agents_run(
    task: str = typer.Argument(..., help="High-level objective for the deep agent."),
    model: Optional[str] = typer.Option(None, "--model", help="Override the default model used by the planner."),
    system_prompt: Optional[str] = typer.Option(
        None,
        "--system-prompt",
        help="Custom system prompt for the agent planner.",
    ),
    no_shell: bool = typer.Option(False, "--no-shell", help="Disable shell command execution."),
    no_python: bool = typer.Option(False, "--no-python", help="Disable Python execution tool."),
    no_tavily: bool = typer.Option(False, "--no-tavily", help="Disable Tavily web search tool."),
    no_dify: bool = typer.Option(False, "--no-dify", help="Disable Dify knowledge base tool."),
    no_document: bool = typer.Option(False, "--no-document", help="Disable document generation tool."),
    no_crossref: bool = typer.Option(False, "--no-crossref", help="Disable Crossref journal works tool."),
    no_openalex: bool = typer.Option(False, "--no-openalex", help="Disable OpenAlex tools."),
    engine: str = typer.Option(
        "langgraph",
        "--engine",
        help="Agent runtime engine (langgraph or deepagents).",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit a machine-readable JSON response."),
) -> None:
    """Run the workspace autonomous agent on a free-form task."""

    try:
        agent = build_workspace_deep_agent(
            model=model,
            include_shell=not no_shell,
            include_python=not no_python,
            include_tavily=not no_tavily,
            include_dify_knowledge=not no_dify,
            include_document_agent=not no_document,
            include_crossref=not no_crossref,
            include_openalex=not no_openalex,
            system_prompt=system_prompt,
            engine=engine,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        response = WorkspaceResponse.error("Failed to initialise deep agent.", errors=(str(exc),))
        _emit_response(response, json_output)
        raise typer.Exit(code=1) from exc

    agent_input = {
        "messages": [HumanMessage(content=task)],
        "iterations": 0,
    }
    result = agent.invoke(agent_input)
    final_message = _extract_final_response(result)
    payload = {"final_response": final_message, "state": result}
    response = WorkspaceResponse.ok(payload=payload, message="Deep agent run completed.")
    _emit_response(response, json_output)

    if not json_output:
        typer.echo("")
        typer.echo(final_message or "(no response)")


@docs_app.command("run")
def docs_run(
    workflow: DocumentWorkflowType = typer.Argument(
        ...,
        case_sensitive=False,
        help="Document workflow key (report, patent_disclosure, plan, project_proposal).",
    ),
    topic: str = typer.Option(..., "--topic", "-t", help="Topic or theme for the document."),
    instructions: Optional[str] = typer.Option(
        None,
        "--instructions",
        "-i",
        help="Additional instructions or constraints to pass to the workflow.",
    ),
    audience: Optional[str] = typer.Option(
        None,
        "--audience",
        "-a",
        help="Intended audience description.",
    ),
    language: str = typer.Option(
        "zh",
        "--language",
        "-l",
        help="Output language (default: zh).",
    ),
    skip_research: bool = typer.Option(
        False,
        "--skip-research",
        help="Disable Tavily web search integration for this run.",
    ),
    search_query: Optional[str] = typer.Option(
        None,
        "--search-query",
        help="Override the default Tavily query (defaults to the topic).",
    ),
    ai_review: bool = typer.Option(
        False,
        "--ai-review",
        help="Run an additional AI review pass after drafting.",
    ),
    temperature: float = typer.Option(
        0.4,
        "--temperature",
        help="Sampling temperature for the language model.",
    ),
    purpose: str = typer.Option(
        "general",
        "--purpose",
        help="Model purpose hint (general, deep_research, creative).",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit a machine-readable JSON response."),
) -> None:
    """Run a document-generation workflow."""

    normalised_purpose = purpose.lower().strip()
    if normalised_purpose not in {"general", "deep_research", "creative"}:
        typer.secho("Invalid --purpose value. Choose from general, deep_research, creative.", fg=typer.colors.RED)
        raise typer.Exit(code=2)
    model_purpose: ModelPurpose = normalised_purpose  # type: ignore[assignment]

    config = DocumentWorkflowConfig(
        workflow=workflow,
        topic=topic,
        instructions=instructions,
        audience=audience,
        language=language,
        include_research=not skip_research,
        search_query=search_query,
        include_ai_review=ai_review,
        temperature=temperature,
        model_purpose=model_purpose,
    )

    try:
        result = run_document_workflow(config)
    except Exception as exc:  # pragma: no cover - defensive fallback
        response = WorkspaceResponse.error("Document workflow failed.", errors=(str(exc),))
        _emit_response(response, json_output)
        raise typer.Exit(code=1) from exc

    response = WorkspaceResponse.ok(payload=result, message="Document workflow completed.")
    _emit_response(response, json_output)

    if not json_output:
        typer.echo("")
        typer.echo("# --- Draft Output ---")
        typer.echo(result.get("draft", ""))
        if ai_review and result.get("ai_review"):
            typer.echo("")
            typer.echo("# --- AI Review ---")
            typer.echo(result.get("ai_review", ""))


# --------------------------------------------------------------------------- Mineru


@app.command("mineru-with-images")
def mineru_with_images(
    file: Path = typer.Argument(
        ...,
        path_type=Path,
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to the PDF document to process.",
    ),
    prompt: Optional[str] = typer.Option(None, "--prompt", help="Optional prompt to steer extraction."),
    chunk_type: bool = typer.Option(
        False,
        "--chunk-type/--no-chunk-type",
        help="Return chunked result by type (Mineru server flag).",
    ),
    return_txt: bool = typer.Option(
        False,
        "--return-txt/--no-return-txt",
        help="Return plain text alongside structured output (Mineru server flag).",
    ),
    pretty: bool = typer.Option(
        False,
        "--pretty/--no-pretty",
        help="Ask Mineru to pretty-print the structured output.",
    ),
    save_to_minio: bool = typer.Option(
        False,
        "--save-to-minio/--no-save-to-minio",
        help="Toggle storing results in MinIO on the server side.",
    ),
    minio_address: Optional[str] = typer.Option(None, "--minio-address", help="MinIO service address."),
    minio_access_key: Optional[str] = typer.Option(None, "--minio-access-key", help="MinIO access key."),
    minio_secret_key: Optional[str] = typer.Option(None, "--minio-secret-key", help="MinIO secret key."),
    minio_bucket: Optional[str] = typer.Option(None, "--minio-bucket", help="Target MinIO bucket name."),
    minio_prefix: Optional[str] = typer.Option(None, "--minio-prefix", help="Key prefix used when saving to MinIO."),
    minio_meta: Optional[str] = typer.Option(None, "--minio-meta", help="Metadata string forwarded to MinIO."),
    provider: Optional[str] = typer.Option(None, "--provider", help="Optional model provider forwarded to Mineru."),
    model: Optional[str] = typer.Option(None, "--model", help="Optional model name forwarded to Mineru."),
    url: Optional[str] = typer.Option(None, "--url", help="Override the Mineru endpoint URL."),
    token: Optional[str] = typer.Option(None, "--token", help="Override the Mineru bearer token."),
    timeout: int = typer.Option(300, "--timeout", help="Timeout in seconds for Mineru requests."),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        path_type=Path,
        help="Optional file path to write the raw Mineru JSON response.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit a machine-readable JSON response."),
) -> None:
    """Call the Mineru PDF image extraction API."""

    try:
        client = MineruClient(api_url_override=url, token_override=token, timeout=float(timeout))
        result = client.recognize_with_images(
            file,
            prompt=prompt,
            chunk_type=chunk_type,
            return_txt=return_txt,
            pretty=pretty,
            save_to_minio=save_to_minio,
            minio_address=minio_address,
            minio_access_key=minio_access_key,
            minio_secret_key=minio_secret_key,
            minio_bucket=minio_bucket,
            minio_prefix=minio_prefix,
            minio_meta=minio_meta,
            provider=provider,
            model=model,
        )
    except MineruClientError as exc:
        response = WorkspaceResponse.error("Mineru request failed.", errors=(str(exc),), source="mineru")
        _emit_response(response, json_output)
        raise typer.Exit(code=1) from exc

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        except OSError as exc:
            response = WorkspaceResponse.error("Mineru request succeeded but writing to disk failed.", errors=(str(exc),), source="mineru")
            _emit_response(response, json_output)
            raise typer.Exit(code=2) from exc

    response = WorkspaceResponse.ok(payload=result, message="Mineru image extraction completed.", source="mineru")
    _emit_response(response, json_output)

    if not json_output:
        typer.echo("")
        typer.echo(f"Endpoint: {result.get('request', {}).get('endpoint')}")
        typer.echo(f"File: {result.get('request', {}).get('file')}")


# --------------------------------------------------------------------------- OpenAlex preparation


@app.command("openalex-fetch")
def openalex_fetch(
    query: Optional[str] = typer.Argument(None, help="Search query to find related papers via OpenAlex."),
    since_year: Optional[int] = typer.Option(2019, "--since-year", help="Only include works published on/after this year."),
    limit: int = typer.Option(20, "--limit", min=1, max=200, help="Maximum number of works to fetch."),
    sample: bool = typer.Option(False, "--sample", help="Use OpenAlex sampling instead of sorted results."),
    sort: str = typer.Option("cited_by_count:desc", "--sort", help="OpenAlex sort expression (default: cited_by_count:desc)."),
    doi: Optional[str] = typer.Option(None, "--doi", help="Fetch a single work by DOI."),
    pdf: Optional[Path] = typer.Option(
        None,
        "--pdf",
        path_type=Path,
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Extract DOI from a PDF and fetch the corresponding OpenAlex record.",
    ),
    download_dir: Optional[Path] = typer.Option(
        None,
        "--download-dir",
        path_type=Path,
        file_okay=False,
        resolve_path=True,
        help="Optional directory to download PDFs when available.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit a machine-readable JSON response."),
) -> None:
    """Fetch OpenAlex works metadata (optionally downloading PDFs for later analysis)."""

    client = OpenAlexClient()
    works: list[Mapping[str, Any]] = []
    download_errors: list[str] = []
    pdf_used: Optional[str] = None
    doi_used: Optional[str] = None

    if pdf:
        pdf_used = str(pdf)
        try:
            doi_used = client.extract_doi_from_pdf(pdf)
        except OpenAlexClientError as exc:
            response = WorkspaceResponse.error("Failed to extract DOI from PDF.", errors=(str(exc),), source="openalex")
            _emit_response(response, json_output)
            raise typer.Exit(code=2) from exc

    if doi or doi_used:
        target_doi = (doi or doi_used or "").strip()
        try:
            work = client.search_by_doi(target_doi)
            work = dict(work)
            if pdf_used:
                work["pdf_path"] = pdf_used
            works = [work]
        except OpenAlexClientError as exc:
            response = WorkspaceResponse.error("OpenAlex DOI lookup failed.", errors=(str(exc),), source="openalex")
            _emit_response(response, json_output)
            raise typer.Exit(code=3) from exc
    else:
        if not query or not query.strip():
            response = WorkspaceResponse.error("Provide either --doi/--pdf or a non-empty query.", source="openalex")
            _emit_response(response, json_output)
            raise typer.Exit(code=4)
        try:
            works = client.search_works(
                query,
                since_year=since_year,
                per_page=limit,
                sample=sample,
                sort=sort,
                fields=(
                    "id",
                    "doi",
                    "title",
                    "publication_year",
                    "cited_by_count",
                    "referenced_works_count",
                    "abstract_inverted_index",
                    "open_access",
                    "primary_location",
                    "locations",
                ),
            )
        except OpenAlexClientError as exc:
            response = WorkspaceResponse.error("OpenAlex query failed.", errors=(str(exc),), source="openalex")
            _emit_response(response, json_output)
            raise typer.Exit(code=1) from exc

    prepared: list[Mapping[str, Any]] = []
    for idx, work in enumerate(works, start=1):
        pdf_url = client.extract_pdf_url(work)
        pdf_path: Optional[str] = work.get("pdf_path") if isinstance(work, Mapping) else None
        if download_dir and pdf_url and not pdf_path:
            slug = make_work_slug(work, fallback=str(idx))
            dest = download_dir / f"{slug}.pdf"
            try:
                client.download_pdf(pdf_url, dest)
                pdf_path = str(dest)
            except OpenAlexClientError as exc:
                download_errors.append(str(exc))
        prepared.append(
            {
                "id": work.get("id"),
                "doi": work.get("doi"),
                "title": work.get("title"),
                "publication_year": work.get("publication_year"),
                "cited_by_count": work.get("cited_by_count"),
                "referenced_works_count": work.get("referenced_works_count"),
                "open_access": work.get("open_access"),
                "abstract_inverted_index": work.get("abstract_inverted_index"),
                "pdf_url": pdf_url,
                "pdf_path": pdf_path,
            }
        )

    payload = {
        "query": query,
        "doi": doi_used or doi,
        "pdf_used": pdf_used,
        "count": len(prepared),
        "works": prepared,
        "download_errors": download_errors or None,
    }
    response = WorkspaceResponse.ok(payload=payload, message="OpenAlex fetch completed.", source="openalex")
    _emit_response(response, json_output)

    if not json_output:
        typer.echo("")
        for item in prepared:
            label = "[pdf]" if item.get("pdf_path") else "[no-pdf]"
            typer.echo(f"{label} {item.get('title') or '(untitled)'} — cites: {item.get('cited_by_count')}")
        if download_errors:
            typer.echo("")
            typer.echo("Download issues:")
            for err in download_errors:
                typer.echo(f"- {err}")


# --------------------------------------------------------------------------- Citation study


@app.command("citation-study")
def citation_study(
    works_file: Optional[Path] = typer.Option(
        None,
        "--works-file",
        path_type=Path,
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Optional JSON file containing works metadata (from another pipeline or custom source).",
    ),
    doi: list[str] = typer.Option(
        [],
        "--doi",
        help="One or more DOIs to score via Supabase (and optionally local PDFs).",
    ),
    pdf_dir: Optional[Path] = typer.Option(
        None,
        "--pdf-dir",
        path_type=Path,
        exists=True,
        file_okay=False,
        resolve_path=True,
        help="Directory containing PDFs to auto-match against works by DOI/ID/title.",
    ),
    mode: str = typer.Option(
        "supabase",
        "--mode",
        help="Fulltext ingestion mode: 'supabase' (use Supabase sci_search) or 'pdf' (extract text from local PDFs).",
        show_default=True,
    ),
    supabase_top_k: int = typer.Option(10, "--supabase-top-k", help="Supabase sci_search topK for fulltext retrieval."),
    supabase_ext_k: int = typer.Option(80, "--supabase-ext-k", help="Supabase sci_search extK for fulltext retrieval."),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="(Deprecated) Override the OpenAI model for LLM scoring; defaults to secrets configuration.",
    ),
    temperature: float = typer.Option(0.2, "--temperature", help="Sampling temperature for LLM scoring."),
    pdf: Optional[Path] = typer.Option(
        None,
        "--pdf",
        path_type=Path,
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Optional PDF file to summarize figures via Mineru and feed into scoring.",
    ),
    use_mineru: bool = typer.Option(False, "--use-mineru", help="Call Mineru to extract figure notes when a PDF is available."),
    mineru_url: Optional[str] = typer.Option(None, "--mineru-url", help="Override Mineru endpoint when using --pdf."),
    mineru_token: Optional[str] = typer.Option(None, "--mineru-token", help="Override Mineru token when using --pdf."),
    mineru_timeout: int = typer.Option(
        300,
        "--mineru-timeout",
        help="Timeout in seconds for Mineru requests (default: 300 to tolerate slow responses).",
    ),
    include_mineru_result: bool = typer.Option(
        False,
        "--include-mineru-result",
        help="Include Mineru raw figure extraction result in the output for investigation.",
    ),
    max_fulltext_chars: int = typer.Option(
        30000,
        "--max-fulltext-chars",
        help="Cap the amount of fulltext (Supabase/PDF) sent to the LLM to avoid overly long prompts.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit a machine-readable JSON response."),
) -> None:
    """Fetch fulltext (Supabase or PDF) and generate a rubric-based citation report."""

    config = CitationStudyConfig(
        works_file=works_file,
        dois=doi,
        pdf_dir=pdf_dir,
        mode=mode,
        supabase_top_k=supabase_top_k,
        supabase_ext_k=supabase_ext_k,
        temperature=temperature,
        pdf=pdf,
        use_mineru=use_mineru,
        mineru_url=mineru_url,
        mineru_token=mineru_token,
        mineru_timeout=mineru_timeout,
        include_mineru_result=include_mineru_result,
        max_fulltext_chars=max_fulltext_chars,
    )

    try:
        payload = run_citation_study(config)
    except ValueError as exc:
        response = WorkspaceResponse.error(str(exc), source="citation-study")
        _emit_response(response, json_output)
        raise typer.Exit(code=2) from exc
    except (SupabaseClientError, MineruClientError) as exc:
        response = WorkspaceResponse.error("Citation study failed.", errors=(str(exc),), source="citation-study")
        _emit_response(response, json_output)
        raise typer.Exit(code=2) from exc

    response = WorkspaceResponse.ok(payload=payload, message="Citation analysis completed.", source="citation-study")
    _emit_response(response, json_output)

    if not json_output:
        works = payload.get("works", []) if isinstance(payload, dict) else []
        if works:
            typer.echo("")
        for idx, item in enumerate(works):
            title = item.get("title") or "(untitled)"
            source_mode = item.get("fulltext_mode") or "n/a"
            typer.echo(f"- [{source_mode}] {title}")
            report = item.get("report") or ""
            if report.strip():
                typer.echo("")
                typer.echo(report.strip())
                typer.echo("")
            if idx != len(works) - 1:
                typer.echo("\n---\n")


# --------------------------------------------------------------------------- Citation impact report (plain text)
@app.command("citation-report")
def citation_report(
    text_file: Path = typer.Argument(
        ...,
        path_type=Path,
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Plain-text paper content (figures already textualised).",
    ),
    title: Optional[str] = typer.Option(None, "--title", help="Optional title override for the report header."),
    temperature: float = typer.Option(0.3, "--temperature", help="Sampling temperature for the LLM."),
    json_output: bool = typer.Option(False, "--json", help="Emit a machine-readable JSON response."),
) -> None:
    """Generate an impact assessment report from a plain-text paper using citation criteria."""

    config = CitationTextReportConfig(text_path=text_file, title=title, temperature=temperature)
    try:
        payload = generate_citation_text_report(config)
    except Exception as exc:
        response = WorkspaceResponse.error("Citation report generation failed.", errors=(str(exc),), source="citation-report")
        _emit_response(response, json_output)
        raise typer.Exit(code=1) from exc

    response = WorkspaceResponse.ok(payload=payload, message="Citation impact report generated.", source="citation-report")
    _emit_response(response, json_output)

    if not json_output:
        typer.echo("")
        typer.echo(payload.get("report", ""))


# --------------------------------------------------------------------------- Journal bands
@app.command("journal-bands-analyze")
def journal_bands_analyze(
    journal_issn: str = typer.Option(..., "--issn", help="Target journal ISSN."),
    journal_name: str = typer.Option(..., "--journal", help="Target journal name for exact matching."),
    publish_year: Optional[int] = typer.Option(None, "--year", help="Optional publication year filter."),
    band_limit: int = typer.Option(200, "--band-limit", min=10, max=400, help="Max papers to include per band after percentile split."),
    max_records: int = typer.Option(800, "--max-records", min=50, max=2000, help="Maximum records to fetch from OpenAlex."),
    top_k: int = typer.Option(10, "--top-k", help="Supabase sci_search topK for fulltext retrieval."),
    ext_k: int = typer.Option(80, "--ext-k", help="Supabase sci_search extK for fulltext retrieval."),
    show_fulltext: bool = typer.Option(False, "--show-fulltext", help="Include retrieved fulltext snippets in the output for debugging."),
    output: Path = typer.Option(
        Path("journal_bands_summary.md"),
        "--output",
        "-o",
        path_type=Path,
        dir_okay=False,
        resolve_path=True,
        help="Path to write the Markdown summary.",
    ),
) -> None:
    """Fetch citation bands for a journal and summarise via Supabase + LLM."""

    config = JournalBandsConfig(
        journal_issn=journal_issn,
        journal_name=journal_name,
        publish_year=publish_year,
        band_limit=band_limit,
        max_records=max_records,
        top_k=top_k,
        ext_k=ext_k,
        show_fulltext=show_fulltext,
        output=output,
    )

    try:
        result_path = run_journal_bands_agent(config, log=lambda msg: typer.echo(f"[journal-bands] {msg}"))
    except SupabaseClientError as exc:
        typer.secho(f"Supabase 获取全文失败: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # pragma: no cover - defensive
        typer.secho(f"Journal bands analysis failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Summary streaming written to {result_path}")


# --------------------------------------------------------------------------- MCP


def _load_mcp_configs() -> Mapping[str, MCPServerSecrets]:
    try:
        secrets = load_secrets()
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=2)
    if not secrets.mcp_servers:
        secrets_path = discover_secrets_path()
        message = f"""No MCP services configured in {secrets_path}. Populate a *_mcp section (see `.sercrets/secrets.example.toml`)."""
        typer.secho(message, fg=typer.colors.YELLOW)
        raise typer.Exit(code=3)
    return secrets.mcp_servers


@mcp_app.command("services")
def list_mcp_services() -> None:
    """List configured MCP services from the secrets file."""

    configs = _load_mcp_configs()
    typer.echo("Configured MCP services:")
    for service in configs.values():
        typer.echo(f"- {service.service_name} ({service.transport}) -> {service.url}")


@mcp_app.command("tools")
def list_mcp_tools(service_name: str) -> None:
    """Enumerate tools exposed by a configured MCP service."""

    configs = _load_mcp_configs()
    if service_name not in configs:
        available = ", ".join(sorted(configs)) or "none"
        typer.secho(f"Service '{service_name}' not found. Available: {available}", fg=typer.colors.RED)
        raise typer.Exit(code=4)

    with MCPToolClient(configs) as client:
        tools = client.list_tools(service_name)

    if not tools:
        typer.echo(f"No tools advertised by service '{service_name}'.")
        return

    typer.echo(f"Tools available on '{service_name}':")
    for tool in tools:
        description = getattr(tool, "description", "") or ""
        if description:
            typer.echo(f"- {tool.name}: {description}")
        else:
            typer.echo(f"- {tool.name}")


@mcp_app.command("invoke")
def invoke_mcp_tool(
    service_name: str,
    tool_name: str,
    args: Optional[str] = typer.Option(None, "--args", "-a", help="JSON object with tool arguments."),
    args_file: Optional[Path] = typer.Option(
        None,
        "--args-file",
        path_type=Path,
        help="Path to a JSON file containing tool arguments.",
    ),
) -> None:
    """Invoke a tool exposed by a configured MCP service."""

    if args and args_file:
        typer.secho("Use either --args or --args-file, not both.", fg=typer.colors.RED)
        raise typer.Exit(code=5)

    payload: Mapping[str, Any] = {}
    if args:
        try:
            payload = json.loads(args)
        except json.JSONDecodeError as exc:
            typer.secho(f"Invalid JSON for --args: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=6) from exc
    elif args_file:
        try:
            payload = json.loads(args_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            typer.secho(f"Invalid JSON in file {args_file}: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=7) from exc
        except OSError as exc:
            typer.secho(f"Failed to read arguments file {args_file}: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=8) from exc

    configs = _load_mcp_configs()
    if service_name not in configs:
        available = ", ".join(sorted(configs)) or "none"
        typer.secho(f"Service '{service_name}' not found. Available: {available}", fg=typer.colors.RED)
        raise typer.Exit(code=9)

    with MCPToolClient(configs) as client:
        result, attachments = client.invoke_tool(service_name, tool_name, payload)

    typer.echo("Tool result:")
    typer.echo(_format_result(result))
    if attachments:
        typer.echo("\nAttachments:")
        for attachment in attachments:
            typer.echo(_format_result(attachment))


def _extract_final_response(result: Any) -> str:
    if isinstance(result, Mapping) and "final_response" in result:
        return str(result["final_response"])
    if isinstance(result, Mapping):
        messages = result.get("messages")
        if isinstance(messages, Sequence) and messages:
            last = messages[-1]
            if isinstance(last, Mapping):
                content = last.get("content")
                if isinstance(content, list):
                    return " ".join(str(chunk) for chunk in content)
                if content is not None:
                    return str(content)
            if hasattr(last, "content"):
                return str(getattr(last, "content"))
        if "response" in result:
            return str(result["response"])
    if hasattr(result, "content"):
        return str(getattr(result, "content"))
    return str(result)


def _emit_response(response: WorkspaceResponse, json_output: bool) -> None:
    if json_output:
        # Write JSON bytes directly to stdout.buffer using UTF-8 to avoid
        # encoding issues when users redirect CLI output to files or pipes.
        try:
            sys.stdout.buffer.write((response.to_json() + "\n").encode("utf-8"))
            sys.stdout.buffer.flush()
            return
        except Exception:
            # Fallback to typer.echo if direct buffer write fails for some reason.
            typer.echo(response.to_json())
            return

    typer.echo(response.message)
    if response.status != "success" and response.errors:
        typer.echo("")
        typer.echo("Errors:")
        for err in response.errors:
            typer.echo(f"- {err}")


@gemini_app.command("deep-research")
def gemini_deep_research(
    prompt: Optional[str] = typer.Argument(
        None,
        help="Research prompt for the Gemini Deep Research agent.",
        metavar="PROMPT",
    ),
    interaction_id: Optional[str] = typer.Option(
        None,
        "--interaction-id",
        help="Existing interaction ID to poll or inspect instead of creating a new task.",
    ),
    agent: Optional[str] = typer.Option(None, "--agent", help="Override the Deep Research agent name."),
    file_search_store: list[str] = typer.Option(
        [],
        "--file-search-store",
        help="Attach File Search store names to expose private data to the agent.",
    ),
    poll: bool = typer.Option(False, "--poll", help="Poll until the interaction completes."),
    poll_interval: float = typer.Option(
        10.0,
        "--poll-interval",
        min=1.0,
        help="Seconds to wait between polling requests (default: 10s).",
    ),
    max_polls: int = typer.Option(
        360,
        "--max-polls",
        min=1,
        help="Maximum number of polls before timing out (default: 360 for ~1 hour at 10s).",
    ),
    thinking_summaries: bool = typer.Option(
        True,
        "--thinking-summaries/--no-thinking-summaries",
        help="Enable thinking summaries in the Deep Research agent config.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit a machine-readable JSON response."),
) -> None:
    """Launch or poll a Gemini Deep Research task using background execution."""

    if not prompt and not interaction_id:
        typer.secho("Provide a PROMPT or --interaction-id to resume an existing task.", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    try:
        client = GeminiDeepResearchClient()
    except GeminiDeepResearchError as exc:
        response = WorkspaceResponse.error("Gemini Deep Research client initialisation failed.", errors=(str(exc),), source="gemini")
        _emit_response(response, json_output)
        raise typer.Exit(code=3)

    started = False
    initial_result: Mapping[str, Any] | None = None
    target_interaction_id = interaction_id
    try:
        if interaction_id:
            initial_result = client.get_interaction(interaction_id)
        else:
            initial_result = client.start_research(
                prompt or "",
                agent=agent,
                file_search_stores=file_search_store or None,
                include_thinking_summaries=thinking_summaries,
            )
            started = True
            target_interaction_id = initial_result.get("interaction_id")
    except GeminiDeepResearchError as exc:
        response = WorkspaceResponse.error("Gemini Deep Research request failed.", errors=(str(exc),), source="gemini")
        _emit_response(response, json_output)
        raise typer.Exit(code=4)

    final_result = None
    if poll:
        if not target_interaction_id:
            response = WorkspaceResponse.error(
                "Cannot poll without an interaction ID.",
                errors=("Interaction ID missing from start response.",),
                source="gemini",
            )
            _emit_response(response, json_output)
            raise typer.Exit(code=5)
        try:
            final_result = client.poll_until_complete(
                target_interaction_id,
                interval=poll_interval,
                max_attempts=max_polls,
            )
        except GeminiDeepResearchError as exc:
            response = WorkspaceResponse.error("Gemini Deep Research polling failed.", errors=(str(exc),), source="gemini")
            _emit_response(response, json_output)
            raise typer.Exit(code=6)

    payload = {
        "interaction": initial_result,
        "interaction_id": target_interaction_id,
        "final_interaction": final_result,
        "polling": poll,
        "started": started,
    }
    message = "Gemini Deep Research run completed." if final_result else "Gemini Deep Research request submitted."
    response = WorkspaceResponse.ok(payload=payload, message=message, source="gemini")
    _emit_response(response, json_output)

    if not json_output:
        typer.echo("")
        typer.echo(f"Interaction ID: {target_interaction_id or '(unknown)'}")
        typer.echo(f"Current status: {initial_result.get('status') if isinstance(initial_result, Mapping) else 'unknown'}")
        if final_result:
            typer.echo(f"Final status: {final_result.get('status')}")
            interaction_body = final_result.get("interaction") if isinstance(final_result, Mapping) else None
            outputs = interaction_body.get("outputs") if isinstance(interaction_body, Mapping) else None
            if outputs:
                typer.echo("Final output:")
                typer.echo(_format_result(outputs[-1]))
            else:
                typer.echo("No outputs returned yet. Inspect the JSON payload for details.")


@app.command()
def research(
    query: str = typer.Argument(..., help="Query string to send to the Tavily MCP service."),
    service_name: str = typer.Option("tavily", "--service", help="MCP service name defined in the secrets file."),
    tool_name: str = typer.Option("search", "--tool-name", help="Tavily tool name to invoke."),
    json_output: bool = typer.Option(False, "--json", help="Emit a machine-readable JSON response."),
) -> None:
    """Run a standalone research query using the Tavily MCP integration."""

    try:
        client = TavilySearchClient(service_name=service_name, tool_name=tool_name)
        result = client.search(query)
    except TavilySearchError as exc:
        response = WorkspaceResponse.error("Research query failed.", errors=(str(exc),))
        _emit_response(response, json_output)
        raise typer.Exit(code=1)

    response = WorkspaceResponse.ok(payload=result, message="Research query completed.")
    _emit_response(response, json_output)

    if not json_output:
        typer.echo("")
        typer.echo("Top-level research result:")
        typer.echo(_format_result(result.get("result")))


@openalex_app.command("work")
def openalex_work(
    doi: str = typer.Argument(..., help="DOI to look up in OpenAlex (with or without https://doi.org/ prefix)."),
    mailto: Optional[str] = typer.Option(None, "--mailto", help="Optional contact email forwarded to OpenAlex."),
    json_output: bool = typer.Option(False, "--json", help="Emit a machine-readable JSON response."),
) -> None:
    """Look up an OpenAlex work record by DOI."""

    try:
        client = OpenAlexClient(mailto=mailto)
        result = client.work_by_doi(doi if doi.lower().startswith("10.") else doi)
    except OpenAlexClientError as exc:
        response = WorkspaceResponse.error("OpenAlex work lookup failed.", errors=(str(exc),), source="openalex")
        _emit_response(response, json_output)
        raise typer.Exit(code=20)

    response = WorkspaceResponse.ok(payload=result, message="OpenAlex work lookup completed.", source="openalex")
    _emit_response(response, json_output)

    if not json_output:
        typer.echo("")
        typer.echo("OpenAlex work:")
        typer.echo(_format_result(result.get("result")))


@openalex_app.command("cited-by")
def openalex_cited_by(
    work_id: str = typer.Argument(..., help="OpenAlex work ID (e.g. W2072484418)."),
    from_publication_date: Optional[str] = typer.Option(None, "--from", help="Filter citing works published on/after this date (YYYY-MM-DD)."),
    to_publication_date: Optional[str] = typer.Option(None, "--to", help="Filter citing works published on/before this date (YYYY-MM-DD)."),
    per_page: Optional[int] = typer.Option(
        200,
        "--per-page",
        min=1,
        max=200,
        help="Number of citing works per page (OpenAlex max 200).",
    ),
    cursor: Optional[str] = typer.Option(None, "--cursor", help="Cursor token for deep pagination (use '*' for first page)."),
    mailto: Optional[str] = typer.Option(None, "--mailto", help="Optional contact email forwarded to OpenAlex."),
    json_output: bool = typer.Option(False, "--json", help="Emit a machine-readable JSON response."),
) -> None:
    """List works citing a given OpenAlex work ID with optional date filtering."""

    try:
        client = OpenAlexClient(mailto=mailto)
        result = client.cited_by(
            work_id,
            from_publication_date=from_publication_date,
            to_publication_date=to_publication_date,
            per_page=per_page,
            cursor=cursor,
        )
    except OpenAlexClientError as exc:
        response = WorkspaceResponse.error("OpenAlex cited-by lookup failed.", errors=(str(exc),), source="openalex")
        _emit_response(response, json_output)
        raise typer.Exit(code=21)

    response = WorkspaceResponse.ok(payload=result, message="OpenAlex cited-by lookup completed.", source="openalex")
    _emit_response(response, json_output)

    if not json_output:
        typer.echo("")
        typer.echo(f"Cited-by count: {result.get('total_count')}")
        typer.echo("OpenAlex response:")
        typer.echo(_format_result(result.get("result")))


@crossref_app.command("journal-works")
def crossref_journal_works(
    issn: str = typer.Argument(..., help="Journal ISSN (e.g. 1234-5678)."),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Optional query string applied to works."),
    filters: Optional[str] = typer.Option(
        None,
        "--filters",
        "-f",
        help="Crossref filters as JSON (object/array) or raw filter string (e.g. from-pub-date:2020-01-01).",
    ),
    sort: Optional[str] = typer.Option(None, "--sort", help="Crossref sort field (score, published, updated, etc.)."),
    order: Optional[str] = typer.Option(None, "--order", help="Sort direction (asc or desc)."),
    rows: Optional[int] = typer.Option(
        None,
        "--rows",
        min=1,
        max=1000,
        help="Maximum results to return (1-1000).",
    ),
    offset: Optional[int] = typer.Option(None, "--offset", min=0, help="Offset for pagination (incompatible with cursor)."),
    cursor: Optional[str] = typer.Option(None, "--cursor", help="Cursor token for deep paging ('*' for first page)."),
    cursor_max: Optional[int] = typer.Option(None, "--cursor-max", min=0, help="Maximum records scanned with cursor."),
    sample: Optional[int] = typer.Option(None, "--sample", min=1, help="Random sample size (cannot combine with cursor)."),
    select: Optional[str] = typer.Option(None, "--select", help="Fields to return (comma-separated or JSON array)."),
    mailto: Optional[str] = typer.Option(None, "--mailto", help="Contact email forwarded to Crossref (recommended)."),
    json_output: bool = typer.Option(False, "--json", help="Emit a machine-readable JSON response."),
) -> None:
    """Call Crossref's `/journals/{issn}/works` endpoint."""

    filter_payload: Any | None = None
    if filters:
        try:
            parsed_filters = json.loads(filters)
        except json.JSONDecodeError:
            filter_payload = filters
        else:
            if isinstance(parsed_filters, (Mapping, list, str)):
                filter_payload = parsed_filters
            else:
                typer.secho("--filters must decode to a JSON object, array, or string.", fg=typer.colors.RED)
                raise typer.Exit(code=17)

    select_payload: Any | None = None
    if select:
        try:
            parsed_select = json.loads(select)
        except json.JSONDecodeError:
            select_payload = select
        else:
            if isinstance(parsed_select, (list, str)):
                select_payload = parsed_select
            else:
                typer.secho("--select must decode to a JSON string or array.", fg=typer.colors.RED)
                raise typer.Exit(code=18)

    try:
        client = CrossrefClient()
        result = client.list_journal_works(
            issn,
            query=query,
            filters=filter_payload,
            sort=sort,
            order=order,
            rows=rows,
            offset=offset,
            cursor=cursor,
            cursor_max=cursor_max,
            sample=sample,
            select=select_payload,
            mailto=mailto,
        )
    except CrossrefClientError as exc:
        response = WorkspaceResponse.error("Crossref query failed.", errors=(str(exc),), source="crossref")
        _emit_response(response, json_output)
        raise typer.Exit(code=19)

    response = WorkspaceResponse.ok(payload=result, message="Crossref journal works query completed.", source="crossref")
    _emit_response(response, json_output)

    if not json_output:
        typer.echo("")
        typer.echo("Crossref response overview:")
        typer.echo(_format_result(result.get("result")))


_DIFY_SEARCH_METHODS = ("hybrid_search", "semantic_search", "full_text_search", "keyword_search")


@knowledge_app.command("retrieve")
def knowledge_retrieve(
    query: str = typer.Argument(..., help="Query string to search within the configured Dify knowledge base."),
    top_k: Optional[int] = typer.Option(
        None,
        "--top-k",
        min=1,
        help="Override the number of chunks returned (defaults to server setting).",
    ),
    search_method: Optional[str] = typer.Option(
        None,
        "--search-method",
        help="Override the retrieval search method (hybrid_search, semantic_search, full_text_search, keyword_search).",
    ),
    reranking_enable: Optional[bool] = typer.Option(
        None,
        "--reranking/--no-reranking",
        help="Toggle the reranking stage for Dify retrieval.",
    ),
    reranking_provider: Optional[str] = typer.Option(
        None,
        "--reranking-provider",
        help="Provider identifier for the reranking model.",
    ),
    reranking_model: Optional[str] = typer.Option(
        None,
        "--reranking-model",
        help="Model name for the reranker.",
    ),
    score_threshold: Optional[float] = typer.Option(
        None,
        "--score-threshold",
        help="Score threshold applied when filtering retrieval results.",
    ),
    score_threshold_enabled: Optional[bool] = typer.Option(
        None,
        "--score-threshold-enabled/--no-score-threshold-enabled",
        help="Explicitly enable or disable score threshold filtering.",
    ),
    weights: Optional[float] = typer.Option(
        None,
        "--semantic-weight",
        help="Semantic weight applied when using hybrid search.",
    ),
    metadata_filters: Optional[str] = typer.Option(
        None,
        "--metadata",
        help="JSON object or array describing metadata filtering conditions.",
    ),
    options: Optional[str] = typer.Option(
        None,
        "--options",
        help="JSON object with additional Dify retrieval parameters (e.g. filters, reranking).",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit a machine-readable JSON response."),
) -> None:
    """Call the Dify knowledge base directly without using MCP."""

    extra_options: Mapping[str, Any] | None = None
    if options:
        try:
            parsed = json.loads(options)
        except json.JSONDecodeError as exc:
            typer.secho(f"Invalid JSON for --options: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=10) from exc
        if not isinstance(parsed, Mapping):
            typer.secho("--options must decode to a JSON object.", fg=typer.colors.RED)
            raise typer.Exit(code=11)
        extra_options = dict(parsed)

    metadata_payload: Any | None = None
    if metadata_filters:
        try:
            parsed_metadata = json.loads(metadata_filters)
        except json.JSONDecodeError as exc:
            typer.secho(f"Invalid JSON for --metadata: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=13) from exc
        if not isinstance(parsed_metadata, (Mapping, list)):
            typer.secho("--metadata must decode to a JSON object or array.", fg=typer.colors.RED)
            raise typer.Exit(code=14)
        metadata_payload = parsed_metadata

    retrieval_overrides: MutableMapping[str, Any] = {}
    if search_method:
        normalized = search_method.strip().lower()
        if normalized not in _DIFY_SEARCH_METHODS:
            typer.secho(
                f"Unsupported search method '{search_method}'. " f"Choose from: {', '.join(_DIFY_SEARCH_METHODS)}.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=15)
        retrieval_overrides["search_method"] = normalized
    if reranking_enable is not None:
        retrieval_overrides["reranking_enable"] = reranking_enable
    if reranking_provider or reranking_model:
        if not reranking_provider or not reranking_model:
            typer.secho("Both --reranking-provider and --reranking-model are required together.", fg=typer.colors.RED)
            raise typer.Exit(code=16)
        retrieval_overrides["reranking_mode"] = {
            "reranking_provider_name": reranking_provider,
            "reranking_model_name": reranking_model,
        }
    if score_threshold_enabled is not None:
        retrieval_overrides["score_threshold_enabled"] = score_threshold_enabled
    if score_threshold is not None:
        retrieval_overrides["score_threshold"] = score_threshold
        retrieval_overrides.setdefault("score_threshold_enabled", True)
    if weights is not None:
        retrieval_overrides["weights"] = weights

    retrieval_payload: Mapping[str, Any] | None = retrieval_overrides or None

    try:
        client = DifyKnowledgeBaseClient()
        result = client.retrieve(
            query,
            top_k=top_k,
            retrieval_model=retrieval_payload,
            metadata_filters=metadata_payload,
            options=extra_options,
        )
    except DifyKnowledgeBaseError as exc:
        response = WorkspaceResponse.error("Knowledge base retrieval failed.", errors=(str(exc),))
        _emit_response(response, json_output)
        raise typer.Exit(code=12) from exc

    response = WorkspaceResponse.ok(payload=result, message="Knowledge base retrieval completed.")
    _emit_response(response, json_output)

    if not json_output:
        typer.echo("")
        typer.echo("Retrieved chunks:")
        typer.echo(_format_result(result.get("result")))


def _format_result(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, indent=2, ensure_ascii=True)
    except TypeError:
        return repr(value)


def main() -> None:
    """Entry point for python -m execution."""
    app()


if __name__ == "__main__":
    main()


def _cli_tool_configs() -> Sequence[CLIToolConfig]:
    return load_workspace_config().cli_tools


@embeddings_app.command("generate")
def embeddings_generate(
    texts: list[str] = typer.Argument(..., help="One or more input texts to embed.", metavar="TEXT"),
    model: str | None = typer.Option(None, "--model", help="Override the embedding model name."),
    encoding_format: str = typer.Option("float", "--encoding-format", help="OpenAI encoding_format parameter."),
    user: str | None = typer.Option(None, "--user", help="Optional user identifier forwarded to the API."),
    json_output: bool = typer.Option(False, "--json", help="Emit a machine-readable JSON response."),
) -> None:
    """Generate embeddings using the configured OpenAI-compatible endpoint."""

    client = OpenAICompatibleEmbeddingClient()
    try:
        result = client.embed(
            texts,
            model_override=model,
            encoding_format=encoding_format,
            user=user,
        )
    except OpenAIEmbeddingError as exc:
        response = WorkspaceResponse.error("Embedding generation failed.", errors=(str(exc),), source="embeddings")
        _emit_response(response, json_output)
        raise typer.Exit(code=1)

    payload = WorkspaceResponse.ok(
        payload={
            "model": result.model,
            "dimensions": result.dimensions,
            "embeddings": result.embeddings,
            "usage": dict(result.usage) if result.usage else None,
            "warnings": list(result.warnings) if result.warnings else None,
        },
        message="Embeddings generated successfully.",
        source="embeddings",
    )
    _emit_response(payload, json_output)

    if not json_output:
        typer.echo("")
        typer.echo(f"Embeddings returned: {len(result.embeddings)} vectors (dimension={result.dimensions or 'unknown'}).")
        if result.warnings:
            typer.echo("Warnings:")
            for note in result.warnings:
                typer.echo(f"- {note}")
