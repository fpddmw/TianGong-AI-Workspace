"""
Workspace configuration loader.

The loader keeps runtime settings (CLI integrations, registered tools, etc.)
outside of the Python modules so users can extend the workspace without
touching code. Configuration currently lives under the ``tool.tiangong``
section inside ``pyproject.toml`` and falls back to built-in defaults when
entries are missing.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
PYPROJECT_PATH = WORKSPACE_ROOT / "pyproject.toml"

__all__ = [
    "CLIToolConfig",
    "RegistryEntryConfig",
    "WorkspaceConfig",
    "load_workspace_config",
]


@dataclass(slots=True, frozen=True)
class CLIToolConfig:
    """Configuration for a CLI integration tracked by the workspace."""

    command: str
    label: str
    version_args: tuple[str, ...] = ("--version",)


@dataclass(slots=True, frozen=True)
class RegistryEntryConfig:
    """Configuration describing a tool entry presented to agents."""

    name: str
    description: str
    category: str
    entrypoint: str
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class WorkspaceConfig:
    """Top-level workspace configuration."""

    cli_tools: tuple[CLIToolConfig, ...]
    registry: tuple[RegistryEntryConfig, ...]


def _project_data() -> Mapping[str, Any]:
    if not PYPROJECT_PATH.exists():
        return {}
    with PYPROJECT_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _get_workspace_section(data: Mapping[str, Any]) -> Mapping[str, Any]:
    tool_section = data.get("tool") or {}
    tiangong_section = tool_section.get("tiangong") or {}
    return tiangong_section.get("workspace") or {}


def _parse_cli_tools(entries: Sequence[Mapping[str, Any]] | None) -> tuple[CLIToolConfig, ...]:
    if not entries:
        return tuple(DEFAULT_CLI_TOOLS)
    configs: list[CLIToolConfig] = []
    for entry in entries:
        version_args = entry.get("version_args") or ["--version"]
        if isinstance(version_args, str):
            version_args = [version_args]
        configs.append(
            CLIToolConfig(
                command=str(entry["command"]),
                label=str(entry["label"]),
                version_args=tuple(str(arg) for arg in version_args),
            )
        )
    return tuple(configs)


def _normalise_tags(tags: Iterable[str] | None) -> tuple[str, ...]:
    if not tags:
        return ()
    return tuple(tag.strip() for tag in tags if tag.strip())


def _parse_registry(entries: Sequence[Mapping[str, Any]] | None) -> tuple[RegistryEntryConfig, ...]:
    if not entries:
        return tuple(DEFAULT_REGISTRY_ENTRIES)
    configs: list[RegistryEntryConfig] = []
    for entry in entries:
        metadata_entry = entry.get("metadata")
        metadata: Mapping[str, Any] | None = None
        if isinstance(metadata_entry, Mapping):
            metadata = dict(metadata_entry)
        configs.append(
            RegistryEntryConfig(
                name=str(entry["name"]),
                description=str(entry["description"]),
                category=str(entry["category"]),
                entrypoint=str(entry["entrypoint"]),
                tags=_normalise_tags(entry.get("tags")),
                metadata=metadata,
            )
        )
    return tuple(configs)


@lru_cache(maxsize=1)
def load_workspace_config() -> WorkspaceConfig:
    """Load workspace configuration from ``pyproject.toml``."""

    data = _project_data()
    workspace_section = _get_workspace_section(data)
    cli_entries = _parse_cli_tools(workspace_section.get("cli_tools"))
    registry_entries = _parse_registry(workspace_section.get("tool_registry"))
    return WorkspaceConfig(cli_tools=cli_entries, registry=registry_entries)


# --------------------------------------------------------------------------- defaults

DEFAULT_CLI_TOOLS: tuple[CLIToolConfig, ...] = (
    CLIToolConfig(command="openai", label="OpenAI CLI (Codex)"),
    CLIToolConfig(command="gcloud", label="Google Cloud CLI (Gemini)"),
    CLIToolConfig(command="claude", label="Claude Code CLI"),
)

DEFAULT_REGISTRY_ENTRIES: tuple[RegistryEntryConfig, ...] = (
    RegistryEntryConfig(
        name="docs.report",
        description="Generate structured business or technical reports using the document workflow agent.",
        category="workflow",
        entrypoint="tiangong_ai_workspace.agents.workflows.run_document_workflow",
        tags=("document", "report"),
    ),
    RegistryEntryConfig(
        name="docs.patent_disclosure",
        description="Draft patent disclosure sheets with optional web research.",
        category="workflow",
        entrypoint="tiangong_ai_workspace.agents.workflows.run_document_workflow",
        tags=("document", "patent"),
    ),
    RegistryEntryConfig(
        name="docs.plan",
        description="Create project or execution plans with milestones and resource breakdowns.",
        category="workflow",
        entrypoint="tiangong_ai_workspace.agents.workflows.run_document_workflow",
        tags=("document", "planning"),
    ),
    RegistryEntryConfig(
        name="docs.project_proposal",
        description="Prepare project proposal drafts optimised for internal reviews.",
        category="workflow",
        entrypoint="tiangong_ai_workspace.agents.workflows.run_document_workflow",
        tags=("document", "proposal"),
    ),
    RegistryEntryConfig(
        name="research.tavily",
        description="Query the Tavily MCP service for live internet research.",
        category="integration",
        entrypoint="tiangong_ai_workspace.tooling.tavily.TavilySearchClient.search",
        tags=("research", "search"),
    ),
    RegistryEntryConfig(
        name="research.crossref_journal_works",
        description="Fetch works for a journal using Crossref's /journals/{issn}/works endpoint.",
        category="integration",
        entrypoint="tiangong_ai_workspace.tooling.crossref.CrossrefClient.list_journal_works",
        tags=("research", "crossref", "literature"),
    ),
    RegistryEntryConfig(
        name="research.openalex_work",
        description="Look up an OpenAlex work record by DOI.",
        category="integration",
        entrypoint="tiangong_ai_workspace.tooling.openalex.OpenAlexClient.work_by_doi",
        tags=("research", "openalex", "literature"),
    ),
    RegistryEntryConfig(
        name="research.openalex_cited_by",
        description="List works citing a given OpenAlex work ID with optional date filtering.",
        category="integration",
        entrypoint="tiangong_ai_workspace.tooling.openalex.OpenAlexClient.cited_by",
        tags=("research", "openalex", "citations"),
    ),
    RegistryEntryConfig(
        name="agents.deep",
        description="Workspace autonomous agent built with LangGraph (shell, Python, Tavily, document workflows).",
        category="agent",
        entrypoint="tiangong_ai_workspace.agents.deep_agent.build_workspace_deep_agent",
        tags=("langgraph", "planner"),
    ),
    RegistryEntryConfig(
        name="runtime.shell",
        description="Shell executor that returns structured stdout/stderr for commands.",
        category="runtime",
        entrypoint="tiangong_ai_workspace.tooling.executors.ShellExecutor.run",
        tags=("shell", "commands"),
    ),
    RegistryEntryConfig(
        name="runtime.python",
        description="Python executor for dynamic scripting with captured stdout/stderr.",
        category="runtime",
        entrypoint="tiangong_ai_workspace.tooling.executors.PythonExecutor.run",
        tags=("python", "scripting"),
    ),
)
