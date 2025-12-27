"""
Utility helpers for Tiangong AI Workspace tooling.

This package exposes lightly opinionated building blocks such as response
schemas, tool registries, and external service wrappers that agents can reuse.
"""

from .crossref import CrossrefClient
from .dify import DifyKnowledgeBaseClient
from .embeddings import OpenAICompatibleEmbeddingClient
from .executors import PythonExecutor, ShellExecutor
from .gemini import GeminiDeepResearchClient
from .get_fulltext import SupabaseClient
from .mineru import MineruClient
from .neo4j import Neo4jClient
from .openalex import OpenAlexClient
from .registry import ToolDescriptor, list_registered_tools
from .responses import ResponsePayload, WorkspaceResponse

__all__ = [
    "CrossrefClient",
    "OpenAlexClient",
    "DifyKnowledgeBaseClient",
    "GeminiDeepResearchClient",
    "OpenAICompatibleEmbeddingClient",
    "OpenAlexClient",
    "MineruClient",
    "PythonExecutor",
    "ResponsePayload",
    "Neo4jClient",
    "SupabaseClient",
    "ShellExecutor",
    "WorkspaceResponse",
    "ToolDescriptor",
    "list_registered_tools",
]
