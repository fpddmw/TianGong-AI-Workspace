"""
Pydantic schemas that describe the workspace tools.

These schemas are shared between LangChain tool definitions and the registry
metadata so the CLI can surface structured information for other agents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, MutableMapping

from pydantic import BaseModel, Field

__all__ = [
    "DocumentToolInput",
    "DocumentToolOutput",
    "DifyKnowledgeBaseInput",
    "DifyKnowledgeBaseOutput",
    "MetadataFilterConditionInput",
    "MetadataFilterGroupInput",
    "RetrievalModelInput",
    "RerankingModeInput",
    "Neo4jCommandInput",
    "Neo4jCommandOutput",
    "PythonCommandInput",
    "PythonCommandOutput",
    "ShellCommandInput",
    "ShellCommandOutput",
    "TavilySearchInput",
    "TavilySearchOutput",
    "descriptor_schema",
]

DocumentWorkflowKey = Literal["report", "patent_disclosure", "plan", "project_proposal"]


class ShellCommandInput(BaseModel):
    command: str = Field(..., description="Shell command to execute within the workspace.")
    timeout: int | None = Field(None, ge=1, description="Optional timeout override in seconds.")


class ShellCommandOutput(BaseModel):
    command: str
    exit_code: int
    stdout: str
    stderr: str
    cwd: str
    duration: float
    timestamp: float


class PythonCommandInput(BaseModel):
    code: str = Field(..., description="Python source code snippet to execute.")


class PythonCommandOutput(BaseModel):
    code: str
    stdout: str
    stderr: str
    duration: float
    timestamp: float
    timed_out: bool


class TavilySearchInput(BaseModel):
    query: str = Field(..., description="Natural language search query.")
    options: dict[str, Any] | None = Field(default=None, description="Optional Tavily MCP parameters.")


class TavilySearchOutput(BaseModel):
    status: Literal["success", "error"]
    data: Mapping[str, Any] | None = None
    message: str | None = None


class RerankingModeInput(BaseModel):
    reranking_provider_name: str = Field(..., description="Provider identifier for the rerank model.")
    reranking_model_name: str = Field(..., description="Model name for the reranker.")


class MetadataFilterConditionInput(BaseModel):
    name: str = Field(..., description="Metadata key to evaluate.")
    comparison_operator: str = Field(..., description="Comparison operator (e.g. eq, ne, contains).")
    value: str | float | int | None = Field(default=None, description="Value to compare against.")


class MetadataFilterGroupInput(BaseModel):
    logical_operator: Literal["and", "or"] = Field(
        "and",
        description="Logical operator used to join metadata filter conditions.",
    )
    conditions: list[MetadataFilterConditionInput] = Field(
        ...,
        min_length=1,
        description="List of metadata filter conditions that must all be satisfied.",
    )


class RetrievalModelInput(BaseModel):
    search_method: Literal["hybrid_search", "semantic_search", "full_text_search", "keyword_search"] | None = Field(
        default=None,
        description="Search strategy to use when retrieving from the knowledge base.",
    )
    reranking_enable: bool | None = Field(
        default=None,
        description="Enable reranking of retrieved segments.",
    )
    reranking_mode: RerankingModeInput | None = Field(
        default=None,
        description="Reranking provider/model configuration.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of segments returned.",
    )
    score_threshold_enabled: bool | None = Field(
        default=None,
        description="Enable score threshold filtering.",
    )
    score_threshold: float | None = Field(
        default=None,
        description="Score threshold applied when filtering retrieval results.",
    )
    weights: float | None = Field(
        default=None,
        description="Weighting applied to semantic search when using hybrid search.",
    )
    metadata_filtering_conditions: MetadataFilterGroupInput | None = Field(
        default=None,
        description="Metadata filters applied to retrieval results.",
    )


class DifyKnowledgeBaseInput(BaseModel):
    query: str = Field(..., description="Knowledge base query string.")
    top_k: int | None = Field(default=None, ge=1, description="Optional limit on retrieved chunks.")
    retrieval_model: RetrievalModelInput | None = Field(
        default=None,
        description="Structured retrieval model overrides forwarded to Dify.",
    )
    metadata_filters: MetadataFilterGroupInput | list[MetadataFilterConditionInput] | None = Field(
        default=None,
        description="Convenience metadata filtering parameters injected into the retrieval model.",
    )
    options: dict[str, Any] | None = Field(
        default=None,
        description="Additional parameters forwarded to the Dify retrieval API.",
    )


class DifyKnowledgeBaseOutput(BaseModel):
    status: Literal["success", "error"]
    data: Mapping[str, Any] | None = None
    message: str | None = None


class Neo4jCommandInput(BaseModel):
    operation: Literal["create", "read", "update", "delete"] = Field(
        "read",
        description="CRUD operation type; determines access mode for the Cypher query.",
    )
    statement: str = Field(..., description="Cypher statement to execute against Neo4j.")
    parameters: Mapping[str, Any] | None = Field(
        default=None,
        description="Optional parameter dictionary passed to the Cypher statement.",
    )
    database: str | None = Field(
        default=None,
        description="Optional database override; falls back to the configured default.",
    )


class Neo4jCommandOutput(BaseModel):
    status: Literal["success", "error"]
    records: list[Mapping[str, Any]] | None = None
    summary: Mapping[str, Any] | None = None
    message: str | None = None


class DocumentToolInput(BaseModel):
    workflow: DocumentWorkflowKey = Field(..., description="Document workflow identifier.")
    topic: str = Field(..., description="Topic or subject for the document.")
    instructions: str | None = Field(default=None, description="Additional authoring instructions.")
    audience: str | None = Field(default=None, description="Intended reader description.")
    language: str = Field(default="zh", description="Output language (default: zh).")
    skip_research: bool = Field(default=False, description="Disable Tavily research stage.")


class DocumentToolOutput(BaseModel):
    status: Literal["success", "error"]
    data: Mapping[str, Any] | None = None
    message: str | None = None


@dataclass(slots=True, frozen=True)
class _SchemaPair:
    input_model: type[BaseModel]
    output_model: type[BaseModel]


_DESCRIPTOR_SCHEMAS: Mapping[str, _SchemaPair] = {
    "runtime.shell": _SchemaPair(ShellCommandInput, ShellCommandOutput),
    "runtime.python": _SchemaPair(PythonCommandInput, PythonCommandOutput),
    "research.tavily": _SchemaPair(TavilySearchInput, TavilySearchOutput),
    "knowledge.dify": _SchemaPair(DifyKnowledgeBaseInput, DifyKnowledgeBaseOutput),
    "database.neo4j": _SchemaPair(Neo4jCommandInput, Neo4jCommandOutput),
    "docs.report": _SchemaPair(DocumentToolInput, DocumentToolOutput),
    "docs.patent_disclosure": _SchemaPair(DocumentToolInput, DocumentToolOutput),
    "docs.plan": _SchemaPair(DocumentToolInput, DocumentToolOutput),
    "docs.project_proposal": _SchemaPair(DocumentToolInput, DocumentToolOutput),
}


def descriptor_schema(descriptor_name: str) -> Mapping[str, Any] | None:
    schema_pair = _DESCRIPTOR_SCHEMAS.get(descriptor_name)
    if not schema_pair:
        return None
    payload: MutableMapping[str, Any] = {
        "input_schema": schema_pair.input_model.model_json_schema(),
        "output_schema": schema_pair.output_model.model_json_schema(),
    }
    return payload
