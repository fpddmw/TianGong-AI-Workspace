"""
Secrets loading utilities for the Tiangong AI Workspace.

This module centralises reading the local `.sercrets/secrets.toml` file so other
packages (such as CLI tools or MCP helpers) can obtain configuration without
needing to know the on-disk layout. The loader mirrors the structure defined in
`.sercrets/secrets.example.toml`.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SECRETS_PATH = Path(os.environ.get("TIANGONG_SECRETS_FILE", WORKSPACE_ROOT / ".sercrets" / "secrets.toml"))


@dataclass(slots=True)
class OpenAISecrets:
    """Secrets required to authenticate with the OpenAI API."""

    api_key: str
    model: Optional[str] = None
    chat_model: Optional[str] = None
    deep_research_model: Optional[str] = None


@dataclass(slots=True)
class OpenAICompatibleEmbeddingSecrets:
    """Configuration for OpenAI-compatible embedding services."""

    url: str
    model: str
    api_key: Optional[str] = None


@dataclass(slots=True)
class MCPServerSecrets:
    """Transport configuration needed to reach an MCP service."""

    service_name: str
    transport: str
    url: str
    api_key: Optional[str] = None
    api_key_header: str = "Authorization"
    api_key_prefix: Optional[str] = None
    timeout: Optional[float] = None

    def connection_payload(self) -> Dict[str, Any]:
        """
        Return keyword arguments expected by `mcp.client.streamable_http.streamablehttp_client`.
        """

        headers: MutableMapping[str, str] = {}
        if self.api_key:
            header_key = self.api_key_header or "Authorization"
            if self.api_key_prefix:
                headers[header_key] = f"{self.api_key_prefix} {self.api_key}"
            else:
                headers[header_key] = self.api_key

        payload: Dict[str, Any] = {"url": self.url}
        if headers:
            payload["headers"] = dict(headers)
        if self.timeout is not None:
            payload["timeout"] = self.timeout
        return payload


@dataclass(slots=True)
class Neo4jSecrets:
    """Credentials required to connect to a Neo4j database."""

    uri: str
    username: str
    password: str
    database: Optional[str] = None


@dataclass(slots=True)
class DifyKnowledgeBaseSecrets:
    """Secrets required to call the Dify knowledge base HTTP API."""

    api_base_url: str
    api_key: str
    dataset_id: str


@dataclass(slots=True)
class MineruSecrets:
    """Configuration for the Mineru document image extraction service."""

    api_url: str
    token: str


@dataclass(slots=True)
class SupabaseSecrets:
    """Configuration for Supabase edge function used for sci_search."""

    api_url: str
    token: str


@dataclass(slots=True)
class Secrets:
    """Container bundling all supported secret entries."""

    openai: Optional[OpenAISecrets]
    mcp_servers: Mapping[str, MCPServerSecrets]
    openai_compatible_embedding: Optional[OpenAICompatibleEmbeddingSecrets] = None
    neo4j: Optional[Neo4jSecrets] = None
    dify_knowledge_base: Optional[DifyKnowledgeBaseSecrets] = None
    mineru: Optional[MineruSecrets] = None
    supabase: Optional[SupabaseSecrets] = None


def discover_secrets_path() -> Path:
    """Return the secrets file path, raising if it does not exist."""

    secrets_path = DEFAULT_SECRETS_PATH
    if not secrets_path.exists():
        message = f"""Unable to locate secrets file at {secrets_path}. Create one based on `.sercrets/secrets.example.toml`."""
        raise FileNotFoundError(message)
    return secrets_path


def load_secrets(path: Optional[Path] = None) -> Secrets:
    """
    Load the secrets file and return strongly typed entries.

    Sections whose table name ends with `_mcp` are interpreted as MCP transport
    settings.
    """

    secrets_path = path or discover_secrets_path()
    with secrets_path.open("rb") as handle:
        data = tomllib.load(handle)

    openai_data = data.get("openai")
    openai_secrets = None
    if isinstance(openai_data, Mapping) and openai_data.get("api_key"):
        openai_secrets = OpenAISecrets(
            api_key=str(openai_data["api_key"]),
            model=_get_opt_str(openai_data, "model"),
            chat_model=_get_opt_str(openai_data, "chat_model"),
            deep_research_model=_get_opt_str(openai_data, "deep_research_model"),
        )

    mcp_entries: Dict[str, MCPServerSecrets] = {}
    for section_name, section in data.items():
        if not section_name.endswith("_mcp"):
            continue
        if not isinstance(section, Mapping):
            continue
        service_name = str(section.get("service_name") or section_name.removesuffix("_mcp"))
        transport = _require_str(section, "transport", section_name)
        url = _require_str(section, "url", section_name)
        mcp_entries[service_name] = MCPServerSecrets(
            service_name=service_name,
            transport=transport,
            url=url,
            api_key=_get_opt_str(section, "api_key"),
            api_key_header=_get_opt_str(section, "api_key_header") or "Authorization",
            api_key_prefix=_get_opt_str(section, "api_key_prefix"),
            timeout=_get_opt_float(section, "timeout"),
        )

    neo4j_data = data.get("neo4j")
    neo4j_secrets = None
    if isinstance(neo4j_data, Mapping):
        uri = _get_opt_str(neo4j_data, "uri")
        username = _get_opt_str(neo4j_data, "username")
        password = _get_opt_str(neo4j_data, "password")
        if uri and username and password:
            neo4j_secrets = Neo4jSecrets(
                uri=uri,
                username=username,
                password=password,
                database=_get_opt_str(neo4j_data, "database"),
            )

    dify_data = data.get("dify_knowledge_base")
    dify_secrets = None
    if isinstance(dify_data, Mapping):
        api_base_url = _get_opt_str(dify_data, "api_base_url")
        api_key = _get_opt_str(dify_data, "api_key")
        dataset_id = _get_opt_str(dify_data, "dataset_id")
        if api_base_url and api_key and dataset_id:
            dify_secrets = DifyKnowledgeBaseSecrets(
                api_base_url=api_base_url.rstrip("/"),
                api_key=api_key,
                dataset_id=dataset_id,
            )

    embedding_data = _load_embedding_section(data)

    mineru_data = data.get("mineru")
    mineru_secrets = None
    if isinstance(mineru_data, Mapping):
        api_url = _get_opt_str(mineru_data, "api_url")
        token = _get_opt_str(mineru_data, "token")
        if api_url and token:
            mineru_secrets = MineruSecrets(api_url=api_url.rstrip("/"), token=token)

    supabase_data = data.get("supabase")
    supabase_secrets = None
    if isinstance(supabase_data, Mapping):
        api_url = _get_opt_str(supabase_data, "api_url")
        token = _get_opt_str(supabase_data, "token")
        if api_url and token:
            supabase_secrets = SupabaseSecrets(api_url=api_url, token=token)

    return Secrets(
        openai=openai_secrets,
        mcp_servers=dict(mcp_entries),
        openai_compatible_embedding=embedding_data,
        neo4j=neo4j_secrets,
        dify_knowledge_base=dify_secrets,
        mineru=mineru_secrets,
        supabase=supabase_secrets,
    )


def _load_embedding_section(data: Mapping[str, Any]) -> Optional[OpenAICompatibleEmbeddingSecrets]:
    for key in ("openai_compatitble_embedding", "openai_compatible_embedding"):
        section = data.get(key)
        if not isinstance(section, Mapping):
            continue
        url = _get_opt_str(section, "url")
        model = _get_opt_str(section, "model")
        if not url or not model:
            continue
        api_key = _get_opt_str(section, "api_key")
        return OpenAICompatibleEmbeddingSecrets(url=url.rstrip("/"), model=model, api_key=api_key)
    return None


def _get_opt_str(container: Mapping[str, Any], key: str) -> Optional[str]:
    value = container.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _get_opt_float(container: Mapping[str, Any], key: str) -> Optional[float]:
    value = container.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected a float-compatible value for '{key}'") from exc


def _require_str(container: Mapping[str, Any], key: str, section_name: str) -> str:
    value = container.get(key)
    if not value:
        raise ValueError(f"Section '{section_name}' must define '{key}'")
    if isinstance(value, str):
        return value
    return str(value)


__all__ = [
    "DEFAULT_SECRETS_PATH",
    "MCPServerSecrets",
    "OpenAICompatibleEmbeddingSecrets",
    "OpenAISecrets",
    "Neo4jSecrets",
    "DifyKnowledgeBaseSecrets",
    "MineruSecrets",
    "Secrets",
    "SupabaseSecrets",
    "discover_secrets_path",
    "load_secrets",
]
