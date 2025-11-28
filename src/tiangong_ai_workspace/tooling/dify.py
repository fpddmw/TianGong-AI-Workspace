"""Dify knowledge base client with structured responses."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Optional, Sequence

import httpx

from ..secrets import DifyKnowledgeBaseSecrets, Secrets, load_secrets

LOGGER = logging.getLogger(__name__)

__all__ = [
    "DifyKnowledgeBaseClient",
    "DifyKnowledgeBaseError",
    "MetadataFilterCondition",
    "MetadataFilterGroup",
    "RetrievalModelConfig",
    "RerankingModeConfig",
]

_SEARCH_METHODS = {
    "hybrid_search",
    "semantic_search",
    "full_text_search",
    "keyword_search",
}

_LOGICAL_OPERATORS = {"and", "or"}


class DifyKnowledgeBaseError(RuntimeError):
    """Raised when the Dify knowledge base API request fails."""


@dataclass(slots=True)
class RerankingModeConfig:
    """Configuration for a reranking provider/model pair."""

    reranking_provider_name: str
    reranking_model_name: str

    def to_payload(self) -> Mapping[str, str]:
        provider = self.reranking_provider_name.strip()
        model = self.reranking_model_name.strip()
        if not provider or not model:
            raise DifyKnowledgeBaseError("Both reranking provider and model names must be provided.")
        return {
            "reranking_provider_name": provider,
            "reranking_model_name": model,
        }


@dataclass(slots=True)
class MetadataFilterCondition:
    """Single metadata filter condition."""

    name: str
    comparison_operator: str
    value: Any = None

    def to_payload(self) -> Mapping[str, Any]:
        field_name = self.name.strip()
        operator = self.comparison_operator.strip()
        if not field_name:
            raise DifyKnowledgeBaseError("Metadata filter condition requires a field name.")
        if not operator:
            raise DifyKnowledgeBaseError("Metadata filter condition requires a comparison operator.")
        payload: MutableMapping[str, Any] = {
            "name": field_name,
            "comparison_operator": operator,
        }
        if self.value is not None:
            payload["value"] = self.value
        return payload


@dataclass(slots=True)
class MetadataFilterGroup:
    """Group of metadata filter conditions combined with AND/OR logic."""

    conditions: Sequence[MetadataFilterCondition]
    logical_operator: str = "and"

    def __post_init__(self) -> None:
        if not self.conditions:
            raise DifyKnowledgeBaseError("Metadata filter group must contain at least one condition.")
        operator = self.logical_operator.lower().strip()
        if operator not in _LOGICAL_OPERATORS:
            raise DifyKnowledgeBaseError("Logical operator must be either 'and' or 'or'.")
        object.__setattr__(self, "logical_operator", operator)
        object.__setattr__(self, "conditions", tuple(self.conditions))

    def to_payload(self) -> Mapping[str, Any]:
        return {
            "logical_operator": self.logical_operator,
            "conditions": [condition.to_payload() for condition in self.conditions],
        }


@dataclass(slots=True)
class RetrievalModelConfig:
    """Strongly typed structure mirroring the Dify retrieval model schema."""

    search_method: Optional[str] = None
    reranking_enable: Optional[bool] = None
    reranking_mode: Optional[RerankingModeConfig] = None
    top_k: Optional[int] = None
    score_threshold_enabled: Optional[bool] = None
    score_threshold: Optional[float] = None
    weights: Optional[float] = None
    metadata_filtering_conditions: Optional[MetadataFilterGroup] = None

    def to_payload(self) -> Mapping[str, Any]:
        payload: MutableMapping[str, Any] = {}
        if self.search_method is not None:
            method = self.search_method.strip().lower()
            if method not in _SEARCH_METHODS:
                raise DifyKnowledgeBaseError(f"Unsupported search method '{self.search_method}'.")
            payload["search_method"] = method
        if self.reranking_enable is not None:
            payload["reranking_enable"] = bool(self.reranking_enable)
        if self.reranking_mode is not None:
            payload["reranking_mode"] = self.reranking_mode.to_payload()
        if self.top_k is not None:
            if self.top_k <= 0:
                raise DifyKnowledgeBaseError("top_k must be greater than zero.")
            payload["top_k"] = int(self.top_k)
        if self.score_threshold_enabled is not None:
            payload["score_threshold_enabled"] = bool(self.score_threshold_enabled)
        if self.score_threshold is not None:
            payload["score_threshold"] = float(self.score_threshold)
        if self.weights is not None:
            payload["weights"] = float(self.weights)
        if self.metadata_filtering_conditions is not None:
            payload["metadata_filtering_conditions"] = self.metadata_filtering_conditions.to_payload()
        return payload


@dataclass(slots=True)
class DifyKnowledgeBaseClient:
    """Lightweight wrapper around the Dify dataset retrieval API."""

    secrets: Optional[Secrets] = None
    timeout: float = 15.0
    http_client: Optional[httpx.Client] = None
    _config: DifyKnowledgeBaseSecrets = field(init=False, repr=False)

    def __post_init__(self) -> None:
        loaded = self.secrets or load_secrets()
        config = loaded.dify_knowledge_base
        if config is None:
            raise DifyKnowledgeBaseError("Dify knowledge base secrets are not configured.")
        object.__setattr__(self, "secrets", loaded)
        object.__setattr__(self, "_config", config)

    def retrieve(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        retrieval_model: RetrievalModelConfig | Mapping[str, Any] | None = None,
        metadata_filters: MetadataFilterGroup | Mapping[str, Any] | Sequence[MetadataFilterCondition | Mapping[str, Any]] | None = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        """Retrieve knowledge chunks for the given query."""

        if not query.strip():
            raise DifyKnowledgeBaseError("Query cannot be empty.")

        payload: MutableMapping[str, Any] = {"query": query}
        if options:
            payload.update(dict(options))

        normalized_filters = _normalize_metadata_filters(metadata_filters)
        existing_model = payload.get("retrieval_model")
        model_payload = _merge_retrieval_model(
            retrieval_model=retrieval_model,
            existing_model=existing_model if isinstance(existing_model, Mapping) else None,
            top_k=top_k,
            metadata_filters=normalized_filters,
        )
        if existing_model is not None and not isinstance(existing_model, Mapping):
            raise DifyKnowledgeBaseError("retrieval_model provided via options must be a JSON object.")
        if model_payload is not None:
            payload["retrieval_model"] = model_payload

        url = f"{self._config.api_base_url}/datasets/{self._config.dataset_id}/retrieve"
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }

        LOGGER.debug("Calling Dify knowledge base: %s", url)
        try:
            response = self._post(url, headers=headers, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            LOGGER.exception("Dify knowledge base request failed")
            raise DifyKnowledgeBaseError(f"HTTP error querying Dify knowledge base: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:  # pragma: no cover - defensive fallback
            raise DifyKnowledgeBaseError("Dify knowledge base returned invalid JSON.") from exc

        return {
            "query": query,
            "result": data,
        }

    def _post(self, url: str, *, headers: Mapping[str, str], json: Mapping[str, Any]) -> httpx.Response:
        if self.http_client is not None:
            return self.http_client.post(url, headers=headers, json=json, timeout=self.timeout)
        return httpx.post(url, headers=headers, json=json, timeout=self.timeout)


def _normalize_metadata_filters(
    filters: MetadataFilterGroup | Mapping[str, Any] | Sequence[MetadataFilterCondition | Mapping[str, Any]] | None,
) -> Mapping[str, Any] | None:
    if filters is None:
        return None
    if isinstance(filters, MetadataFilterGroup):
        return filters.to_payload()
    if isinstance(filters, Mapping):
        if "conditions" in filters:
            conditions = _normalize_filter_conditions(filters["conditions"])
        elif {"name", "comparison_operator"}.issubset(filters.keys()):
            conditions = [_condition_from_mapping(filters)]
        else:
            raise DifyKnowledgeBaseError("metadata_filters mapping must include either 'conditions' or a single condition definition.")
        logical_operator = filters.get("logical_operator", "and")
        return {
            "logical_operator": _normalize_logical_operator(logical_operator),
            "conditions": conditions,
        }
    if isinstance(filters, Sequence) and not isinstance(filters, (str, bytes)):
        conditions = _normalize_filter_conditions(filters)
        return {
            "logical_operator": "and",
            "conditions": conditions,
        }
    raise DifyKnowledgeBaseError("metadata_filters must be a MetadataFilterGroup, mapping, or a sequence of condition mappings.")


def _normalize_filter_conditions(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DifyKnowledgeBaseError("Filter conditions must be provided as a list of objects.")
    conditions: list[Mapping[str, Any]] = []
    for entry in value:
        if isinstance(entry, MetadataFilterCondition):
            conditions.append(entry.to_payload())
            continue
        if not isinstance(entry, Mapping):
            raise DifyKnowledgeBaseError("Each metadata filter condition must be a mapping.")
        conditions.append(_condition_from_mapping(entry))
    if not conditions:
        raise DifyKnowledgeBaseError("metadata_filters must include at least one condition.")
    return conditions


def _condition_from_mapping(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    name = str(entry.get("name") or "").strip()
    comparison_operator = str(entry.get("comparison_operator") or "").strip()
    if not name:
        raise DifyKnowledgeBaseError("Metadata filter condition requires 'name'.")
    if not comparison_operator:
        raise DifyKnowledgeBaseError("Metadata filter condition requires 'comparison_operator'.")
    payload: MutableMapping[str, Any] = {
        "name": name,
        "comparison_operator": comparison_operator,
    }
    if "value" in entry:
        payload["value"] = entry.get("value")
    return payload


def _normalize_logical_operator(value: Any) -> str:
    candidate = str(value or "and").strip().lower()
    if candidate not in _LOGICAL_OPERATORS:
        raise DifyKnowledgeBaseError("metadata_filters logical operator must be 'and' or 'or'.")
    return candidate


def _merge_retrieval_model(
    *,
    retrieval_model: RetrievalModelConfig | Mapping[str, Any] | None,
    existing_model: Mapping[str, Any] | None,
    top_k: Optional[int],
    metadata_filters: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    merged: MutableMapping[str, Any] = {}
    if existing_model:
        merged.update(existing_model)
    if isinstance(retrieval_model, RetrievalModelConfig):
        merged.update(retrieval_model.to_payload())
    elif isinstance(retrieval_model, Mapping):
        merged.update(_strip_none(retrieval_model))
    elif retrieval_model is not None:
        raise DifyKnowledgeBaseError("retrieval_model must be a mapping or RetrievalModelConfig instance.")

    if top_k is not None:
        if top_k <= 0:
            raise DifyKnowledgeBaseError("top_k must be greater than zero.")
        merged["top_k"] = int(top_k)

    if metadata_filters is not None:
        merged["metadata_filtering_conditions"] = metadata_filters

    return dict(merged) if merged else None


def _strip_none(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}
