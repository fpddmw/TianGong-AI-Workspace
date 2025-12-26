from __future__ import annotations

import json
from typing import Any, Mapping

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable

from tiangong_ai_workspace.agents.deep_agent import build_workspace_deep_agent
from tiangong_ai_workspace.secrets import DifyKnowledgeBaseSecrets, GeminiSecrets, MCPServerSecrets, Neo4jSecrets, Secrets
from tiangong_ai_workspace.tooling import (
    GeminiDeepResearchClient,
    PythonExecutor,
    ShellExecutor,
    WorkspaceResponse,
    list_registered_tools,
)
from tiangong_ai_workspace.tooling.crossref import CrossrefClient, CrossrefClientError
from tiangong_ai_workspace.tooling.dify import DifyKnowledgeBaseClient, DifyKnowledgeBaseError
from tiangong_ai_workspace.tooling.gemini import GeminiDeepResearchError
from tiangong_ai_workspace.tooling.neo4j import Neo4jClient, Neo4jToolError
from tiangong_ai_workspace.tooling.openalex import OpenAlexClient
from tiangong_ai_workspace.tooling.tavily import TavilySearchClient, TavilySearchError


def test_workspace_response_json_roundtrip() -> None:
    response = WorkspaceResponse.ok(payload={"value": 42}, message="All good", request_id="abc123")
    payload = json.loads(response.to_json())
    assert payload["status"] == "success"
    assert payload["payload"] == {"value": 42}
    assert payload["metadata"]["request_id"] == "abc123"


def test_tool_registry_contains_core_workflows() -> None:
    registry = list_registered_tools()
    assert "docs.report" in registry
    assert registry["docs.report"].category == "workflow"
    assert "agents.deep" in registry
    assert "runtime.shell" in registry
    assert "runtime.python" in registry
    assert "embeddings.openai_compatible" in registry
    assert "research.crossref_journal_works" in registry
    assert "research.openalex_work" in registry
    assert "research.openalex_cited_by" in registry
    assert "research.gemini_deep_research" in registry


def test_tavily_client_missing_service_raises() -> None:
    secrets = Secrets(openai=None, mcp_servers={})
    with pytest.raises(TavilySearchError):
        TavilySearchClient(secrets=secrets)


def test_tavily_client_custom_service_is_loaded() -> None:
    secrets = Secrets(
        openai=None,
        mcp_servers={
            "custom": MCPServerSecrets(
                service_name="custom",
                transport="streamable_http",
                url="https://example.com",
            )
        },
    )
    client = TavilySearchClient(secrets=secrets, service_name="custom")
    assert client.service_name == "custom"


def test_dify_client_missing_configuration_raises() -> None:
    secrets = Secrets(openai=None, mcp_servers={})
    with pytest.raises(DifyKnowledgeBaseError):
        DifyKnowledgeBaseClient(secrets=secrets)


def test_dify_client_retrieve(monkeypatch: pytest.MonkeyPatch) -> None:
    config = DifyKnowledgeBaseSecrets(
        api_base_url="https://example.com/v1",
        api_key="dataset-123",
        dataset_id="abc",
    )
    secrets = Secrets(openai=None, mcp_servers={}, dify_knowledge_base=config)
    client = DifyKnowledgeBaseClient(secrets=secrets)

    captured: dict[str, Any] = {}

    class _StubResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Mapping[str, Any]:
            return {"chunks": ["A"], "hit": True}

    def fake_post(self, url: str, *, headers: Mapping[str, str], json: Mapping[str, Any]):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return _StubResponse()

    monkeypatch.setattr(DifyKnowledgeBaseClient, "_post", fake_post, raising=False)

    result = client.retrieve(
        "test",
        top_k=3,
        retrieval_model={"search_method": "semantic_search"},
        metadata_filters=[{"name": "tag", "comparison_operator": "eq", "value": "workspace"}],
        options={"retrieval_model": {"reranking_enable": True}},
    )

    assert result["query"] == "test"
    assert result["result"]["hit"]
    assert captured["json"]["retrieval_model"]["top_k"] == 3
    assert captured["json"]["retrieval_model"]["search_method"] == "semantic_search"
    assert captured["json"]["retrieval_model"]["reranking_enable"] is True
    filters = captured["json"]["retrieval_model"]["metadata_filtering_conditions"]
    assert filters["logical_operator"] == "and"
    assert filters["conditions"][0]["name"] == "tag"
    assert captured["headers"]["Authorization"] == "Bearer dataset-123"


def test_crossref_client_list_journal_works(monkeypatch: pytest.MonkeyPatch) -> None:
    client = CrossrefClient(timeout=1.0)
    captured: dict[str, Any] = {}

    class _StubResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Mapping[str, Any]:
            return {"message": {"items": [{"title": ["Example"]}], "total-results": 1}}

    def fake_get(self, url: str, *, params: Mapping[str, Any], headers: Mapping[str, str]):
        captured["url"] = url
        captured["params"] = params
        captured["headers"] = headers
        return _StubResponse()

    monkeypatch.setattr(CrossrefClient, "_get", fake_get, raising=False)

    result = client.list_journal_works(
        "1234-5678",
        query="ai",
        filters={"from-pub-date": "2020-01-01", "until-pub-date": "2020-12-31"},
        sort="published",
        order="asc",
        rows=5,
        select=["title", "DOI"],
        mailto="test@example.com",
    )

    assert captured["url"].endswith("/journals/1234-5678/works")
    assert captured["params"]["query"] == "ai"
    assert "from-pub-date:2020-01-01" in captured["params"]["filter"]
    assert captured["params"]["order"] == "asc"
    assert captured["params"]["rows"] == 5
    assert captured["params"]["select"] == "title,DOI"
    assert captured["params"]["mailto"] == "test@example.com"
    assert result["issn"] == "1234-5678"
    assert result["result"]["message"]["total-results"] == 1


def test_crossref_client_rejects_offset_and_cursor() -> None:
    client = CrossrefClient()
    with pytest.raises(CrossrefClientError):
        client.list_journal_works("1234-5678", offset=1, cursor="*")


def test_openalex_client_work_by_doi(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OpenAlexClient(timeout=1.0)
    captured: dict[str, Any] = {}

    class _StubResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Mapping[str, Any]:
            return {"id": "https://openalex.org/W123"}

    def fake_get(self, url: str, *, params: Mapping[str, Any]):
        captured["url"] = url
        captured["params"] = params
        return _StubResponse()

    monkeypatch.setattr(OpenAlexClient, "_get", fake_get, raising=False)
    result = client.work_by_doi("10.1234/example", mailto="a@b.com")
    assert "example" in captured["url"]
    assert captured["params"]["mailto"] == "a@b.com"
    assert result["result"]["id"] == "https://openalex.org/W123"


def test_openalex_client_cited_by(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OpenAlexClient(timeout=1.0)
    captured: dict[str, Any] = {}

    class _StubResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Mapping[str, Any]:
            return {"meta": {"count": 2}, "results": [{"id": 1}]}

    def fake_get(self, url: str, *, params: Mapping[str, Any]):
        captured["url"] = url
        captured["params"] = params
        return _StubResponse()

    monkeypatch.setattr(OpenAlexClient, "_get", fake_get, raising=False)
    result = client.cited_by(
        "W123",
        from_publication_date="2020-01-01",
        to_publication_date="2021-01-01",
        per_page=100,
        cursor="*",
        mailto="a@b.com",
    )
    assert "cites:W123" in captured["params"]["filter"]
    assert "from_publication_date:2020-01-01" in captured["params"]["filter"]
    assert captured["params"]["per-page"] == 100
    assert captured["params"]["cursor"] == "*"
    assert captured["params"]["mailto"] == "a@b.com"
    assert result["total_count"] == 2


def test_gemini_client_missing_configuration_raises() -> None:
    secrets = Secrets(openai=None, mcp_servers={})
    with pytest.raises(GeminiDeepResearchError):
        GeminiDeepResearchClient(secrets=secrets)


def test_gemini_client_start_research_builds_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    config = GeminiSecrets(api_key="key-123", agent="deep-research-pro-preview-12-2025")
    secrets = Secrets(openai=None, mcp_servers={}, gemini=config)
    client = GeminiDeepResearchClient(secrets=secrets)

    captured: dict[str, Any] = {}

    class _StubResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Mapping[str, Any]:
            return {"id": "abc123", "status": "in_progress", "outputs": []}

    def fake_post(self, url: str, *, headers: Mapping[str, str], json: Mapping[str, Any]):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return _StubResponse()

    monkeypatch.setattr(GeminiDeepResearchClient, "_post", fake_post, raising=False)
    result = client.start_research("Test prompt", file_search_stores=["storeA"])

    assert captured["headers"]["x-goog-api-key"] == "key-123"
    assert captured["json"]["background"] is True
    assert captured["json"]["store"] is True
    assert captured["json"]["tools"][0]["file_search_store_names"] == ["storeA"]
    assert captured["json"]["agent_config"]["thinking_summaries"] == "auto"
    assert result["interaction_id"] == "abc123"


def test_gemini_client_poll_until_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    config = GeminiSecrets(api_key="key-123", agent="deep-research-pro-preview-12-2025")
    secrets = Secrets(openai=None, mcp_servers={}, gemini=config)
    client = GeminiDeepResearchClient(secrets=secrets)
    calls: list[str] = []

    def fake_get(self, interaction_id: str):
        calls.append(interaction_id)
        if len(calls) < 2:
            return {"interaction_id": interaction_id, "status": "in_progress"}
        return {"interaction_id": interaction_id, "status": "completed", "interaction": {"outputs": [{"text": "done"}]}}

    monkeypatch.setattr(GeminiDeepResearchClient, "get_interaction", fake_get, raising=False)
    monkeypatch.setattr("tiangong_ai_workspace.tooling.gemini.time.sleep", lambda _: None)

    result = client.poll_until_complete("abc", interval=0.0, max_attempts=3)
    assert result["status"] == "completed"
    assert calls == ["abc", "abc"]


def test_gemini_client_poll_until_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    config = GeminiSecrets(api_key="key-123", agent="deep-research-pro-preview-12-2025")
    secrets = Secrets(openai=None, mcp_servers={}, gemini=config)
    client = GeminiDeepResearchClient(secrets=secrets)

    def fake_get(self, interaction_id: str):
        return {"interaction_id": interaction_id, "status": "failed", "interaction": {"error": "boom"}}

    monkeypatch.setattr(GeminiDeepResearchClient, "get_interaction", fake_get, raising=False)
    monkeypatch.setattr("tiangong_ai_workspace.tooling.gemini.time.sleep", lambda _: None)

    with pytest.raises(GeminiDeepResearchError):
        client.poll_until_complete("abc", interval=0.0, max_attempts=2)


def test_shell_executor_runs_command() -> None:
    executor = ShellExecutor()
    result = executor.run("echo hello")
    assert result.exit_code == 0
    assert "hello" in result.stdout.lower()


def test_python_executor_captures_output() -> None:
    executor = PythonExecutor()
    result = executor.run("print('hi')")
    assert "hi" in result.stdout
    assert result.stderr == ""


def test_neo4j_client_executes_with_stub_driver() -> None:
    stub_result = _StubNeo4jResult([{"name": "workspace"}])
    stub_driver = _StubNeo4jDriver(stub_result)
    config = Neo4jSecrets(uri="bolt://localhost:7687", username="neo4j", password="pass", database="neo4j")
    client = Neo4jClient(config=config, driver=stub_driver)

    payload = client.execute("MATCH (n) RETURN n", operation="read", parameters={"limit": 1})

    assert payload["records"] == [{"name": "workspace"}]
    assert payload["summary"]["database"] == "neo4j"
    assert payload["summary"]["counters"]["nodes_created"] == 1


def test_neo4j_client_without_configuration_raises() -> None:
    with pytest.raises(Neo4jToolError):
        Neo4jClient(secrets=Secrets(openai=None, mcp_servers={}, neo4j=None))


class StubPlanner(Runnable):
    """Deterministic planner used for testing the workspace agent."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses

    def invoke(self, _: Any, config: Any | None = None) -> str:  # type: ignore[override]
        if not self._responses:
            raise RuntimeError("StubPlanner has no responses left")
        return self._responses.pop(0)


def test_build_workspace_deep_agent_runs_to_completion() -> None:
    planner = StubPlanner(
        [
            '{"thought": "All done.", "action": "finish", "final_response": "Completed task."}',
        ]
    )
    agent = build_workspace_deep_agent(
        llm=planner,
        include_shell=False,
        include_python=False,
        include_tavily=False,
        include_document_agent=False,
    )
    result = agent.invoke({"messages": [HumanMessage(content="Test task")], "iterations": 0})
    assert result["final_response"] == "Completed task."


class DummyChatModel(BaseChatModel):
    """Minimal BaseChatModel implementation for deepagents engine tests."""

    @property
    def _llm_type(self) -> str:
        return "dummy"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:  # type: ignore[override]
        generation = ChatGeneration(message=AIMessage(content="ok"))
        return ChatResult(generations=[generation])


def test_build_workspace_deep_agent_deepagents_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_model = DummyChatModel()
    stub_agent = object()

    def fake_create_deep_agent(*args, **kwargs):
        fake_create_deep_agent.called = True  # type: ignore[attr-defined]
        fake_create_deep_agent.kwargs = kwargs  # type: ignore[attr-defined]
        return stub_agent

    monkeypatch.setattr("tiangong_ai_workspace.agents.deep_agent.create_deep_agent", fake_create_deep_agent)
    agent = build_workspace_deep_agent(
        llm=dummy_model,
        include_shell=False,
        include_python=False,
        include_tavily=False,
        include_document_agent=False,
        engine="deepagents",
    )
    assert agent is stub_agent
    assert getattr(fake_create_deep_agent, "called", False)
    assert "tools" in getattr(fake_create_deep_agent, "kwargs", {})


class _StubNeo4jRecord:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self._payload = dict(payload)

    def data(self) -> Mapping[str, Any]:
        return dict(self._payload)


class _StubCounters:
    nodes_created = 1
    contains_updates = True

    def ignored_method(self) -> None:  # pragma: no cover - ensures callable is skipped
        return None


class _StubSummary:
    def __init__(self) -> None:
        self.query = type("Q", (), {"text": "MATCH ()"})()
        self.database = "neo4j"
        self.query_type = "r"
        self.result_available_after = 1
        self.result_consumed_after = 2
        self.counters = _StubCounters()


class _StubNeo4jResult:
    def __init__(self, records: list[Mapping[str, Any]]) -> None:
        self._records = [_StubNeo4jRecord(record) for record in records]

    def __iter__(self):  # type: ignore[override]
        return iter(self._records)

    def consume(self) -> _StubSummary:
        return _StubSummary()


class _StubNeo4jSession:
    def __init__(self, result: _StubNeo4jResult) -> None:
        self._result = result

    def run(self, statement: str, parameters: Mapping[str, Any] | None = None) -> _StubNeo4jResult:  # noqa: D401
        self.statement = statement
        self.parameters = parameters or {}
        return self._result

    def __enter__(self) -> "_StubNeo4jSession":
        return self

    def __exit__(self, *args: Any) -> None:  # pragma: no cover - no cleanup
        return None


class _StubNeo4jDriver:
    def __init__(self, result: _StubNeo4jResult) -> None:
        self._result = result

    def session(self, **kwargs: Any) -> _StubNeo4jSession:
        self.session_kwargs = kwargs
        return _StubNeo4jSession(self._result)

    def close(self) -> None:  # pragma: no cover - no-op
        return None
