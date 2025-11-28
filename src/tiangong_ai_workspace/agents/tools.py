"""
LangChain tool definitions that expose workspace capabilities to agents.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from langchain_core.tools import tool
from pydantic import BaseModel

from ..tooling.dify import DifyKnowledgeBaseClient, DifyKnowledgeBaseError
from ..tooling.executors import PythonExecutor, ShellExecutor
from ..tooling.neo4j import Neo4jClient, Neo4jToolError
from ..tooling.tavily import TavilySearchClient, TavilySearchError
from ..tooling.tool_schemas import (
    DifyKnowledgeBaseInput,
    DifyKnowledgeBaseOutput,
    DocumentToolInput,
    DocumentToolOutput,
    MetadataFilterConditionInput,
    MetadataFilterGroupInput,
    Neo4jCommandInput,
    Neo4jCommandOutput,
    PythonCommandInput,
    PythonCommandOutput,
    RetrievalModelInput,
    ShellCommandInput,
    ShellCommandOutput,
    TavilySearchInput,
    TavilySearchOutput,
)
from .workflows import DocumentWorkflowConfig, DocumentWorkflowType, run_document_workflow

__all__ = [
    "create_document_tool",
    "create_dify_knowledge_tool",
    "create_neo4j_tool",
    "create_python_tool",
    "create_shell_tool",
    "create_tavily_tool",
]


def create_shell_tool(executor: Optional[ShellExecutor] = None, *, name: str = "run_shell") -> Any:
    exec_instance = executor or ShellExecutor()

    @tool(name, args_schema=ShellCommandInput)
    def run_shell(command: str, timeout: int | None = None) -> Mapping[str, Any]:
        """Execute a shell command inside the workspace environment."""

        result = exec_instance.run(command, timeout=timeout)
        payload = ShellCommandOutput(**result.to_dict())
        return payload.model_dump()

    return run_shell


def create_python_tool(executor: Optional[PythonExecutor] = None, *, name: str = "run_python") -> Any:
    exec_instance = executor or PythonExecutor()

    @tool(name, args_schema=PythonCommandInput)
    def run_python(code: str) -> Mapping[str, Any]:
        """Execute Python code using the shared workspace interpreter."""

        result = exec_instance.run(code)
        payload = PythonCommandOutput(**result.to_dict())
        return payload.model_dump()

    return run_python


def create_tavily_tool(client: Optional[TavilySearchClient] = None, *, name: str = "tavily_search") -> Any:
    tavily_client = client or TavilySearchClient()

    @tool(name, args_schema=TavilySearchInput)
    def tavily_search(query: str, options: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
        """Search the internet using the configured Tavily MCP service."""

        try:
            result = tavily_client.search(query, options=dict(options or {}))
        except TavilySearchError as exc:
            payload = TavilySearchOutput(status="error", message=str(exc))
            return payload.model_dump()
        payload = TavilySearchOutput(status="success", data=result)
        return payload.model_dump()

    return tavily_search


def create_dify_knowledge_tool(client: Optional[DifyKnowledgeBaseClient] = None, *, name: str = "dify_knowledge") -> Any:
    kb_client = client or DifyKnowledgeBaseClient()

    def _dump_model(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(exclude_none=True)
        return value

    def _prepare_metadata_filters(
        filters: MetadataFilterGroupInput | Sequence[MetadataFilterConditionInput] | Mapping[str, Any] | None,
    ) -> Any:
        if filters is None:
            return None
        if isinstance(filters, MetadataFilterGroupInput):
            return filters.model_dump(exclude_none=True)
        if isinstance(filters, Sequence) and not isinstance(filters, (str, bytes)):
            return [_dump_model(item) for item in filters]
        if isinstance(filters, Mapping):
            return dict(filters)
        return filters

    @tool(name, args_schema=DifyKnowledgeBaseInput)
    def dify_knowledge(
        query: str,
        top_k: int | None = None,
        retrieval_model: RetrievalModelInput | Mapping[str, Any] | None = None,
        metadata_filters: MetadataFilterGroupInput | Sequence[MetadataFilterConditionInput] | None = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        """Retrieve chunks from the configured Dify knowledge base."""

        retrieval_payload = _dump_model(retrieval_model) if retrieval_model is not None else None
        metadata_payload = _prepare_metadata_filters(metadata_filters)

        try:
            result = kb_client.retrieve(
                query,
                top_k=top_k,
                retrieval_model=retrieval_payload,
                metadata_filters=metadata_payload,
                options=dict(options or {}),
            )
        except DifyKnowledgeBaseError as exc:
            payload = DifyKnowledgeBaseOutput(status="error", message=str(exc))
            return payload.model_dump()

        payload = DifyKnowledgeBaseOutput(status="success", data=result)
        return payload.model_dump()

    return dify_knowledge


def create_neo4j_tool(client: Optional[Neo4jClient] = None, *, name: str = "run_neo4j_query") -> Any:
    neo4j_client = client or Neo4jClient()

    @tool(name, args_schema=Neo4jCommandInput)
    def run_neo4j_query(
        statement: str,
        operation: str = "read",
        parameters: Optional[Mapping[str, Any]] = None,
        database: str | None = None,
    ) -> Mapping[str, Any]:
        """Execute a Cypher statement against the configured Neo4j database."""

        try:
            result = neo4j_client.execute(statement, parameters=parameters, operation=operation, database=database)
        except Neo4jToolError as exc:
            payload = Neo4jCommandOutput(status="error", message=str(exc))
            return payload.model_dump()

        payload = Neo4jCommandOutput(status="success", records=result.get("records"), summary=result.get("summary"))
        return payload.model_dump()

    return run_neo4j_query


def create_document_tool(*, name: str = "generate_document") -> Any:
    @tool(name, args_schema=DocumentToolInput)
    def generate_document(
        workflow: str,
        topic: str,
        instructions: str | None = None,
        audience: str | None = None,
        language: str = "zh",
        skip_research: bool = False,
    ) -> Mapping[str, Any]:
        """Generate a structured document using the LangGraph workflow."""

        try:
            workflow_type = DocumentWorkflowType(workflow)
        except ValueError:
            payload = DocumentToolOutput(status="error", message=f"Unsupported workflow '{workflow}'.")
            return payload.model_dump()

        config = DocumentWorkflowConfig(
            workflow=workflow_type,
            topic=topic,
            instructions=instructions,
            audience=audience,
            language=language,
            include_research=not skip_research,
        )
        result = run_document_workflow(config)
        payload = DocumentToolOutput(status="success", data=result)
        return payload.model_dump()

    return generate_document
