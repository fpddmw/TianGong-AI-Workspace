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
from .agents.deep_agent import build_workspace_deep_agent
from .mcp_client import MCPToolClient
from .secrets import MCPServerSecrets, discover_secrets_path, load_secrets
from .tooling import WorkspaceResponse, list_registered_tools
from .tooling.config import CLIToolConfig, load_workspace_config
from .tooling.dify import DifyKnowledgeBaseClient, DifyKnowledgeBaseError
from .tooling.embeddings import OpenAICompatibleEmbeddingClient, OpenAIEmbeddingError
from .tooling.llm import ModelPurpose
from .tooling.mineru import MineruClient, MineruClientError
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
        client = MineruClient(api_url_override=url, token_override=token)
        result = client.recognize_with_images(
            file,
            prompt=prompt,
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
        typer.echo(response.to_json())
        return

    typer.echo(response.message)
    if response.status != "success" and response.errors:
        typer.echo("")
        typer.echo("Errors:")
        for err in response.errors:
            typer.echo(f"- {err}")


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
