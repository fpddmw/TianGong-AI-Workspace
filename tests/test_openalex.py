from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from tiangong_ai_workspace.tooling.openalex import LLMCitationAssessor, OpenAlexClient, OpenAlexClientError


def test_search_works_empty_query_raises() -> None:
    client = OpenAlexClient()
    with pytest.raises(OpenAlexClientError):
        client.search_works("")


def test_search_works_parses_results(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OpenAlexClient()

    class _StubResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"results": [{"id": "W1"}]}

    def fake_get(self, url: str, params):
        assert "search" in params
        return _StubResponse()

    monkeypatch.setattr(OpenAlexClient, "_get", fake_get, raising=False)
    results = client.search_works("ai", since_year=2020, per_page=5)
    assert results[0]["id"] == "W1"


def test_classify_work_returns_category() -> None:
    client = OpenAlexClient()
    work = {
        "id": "W2",
        "title": "Sample paper",
        "publication_year": 2018,
        "cited_by_count": 120,
        "referenced_works_count": 40,
        "abstract_inverted_index": {"hello": [0], "world": [1]},
        "open_access": {"is_oa": True},
    }
    result = client.classify_work(work)
    assert result["category"] == "high"
    assert result["score"] > 0


def test_llm_assessor_parses_json(monkeypatch: pytest.MonkeyPatch) -> None:
    work = {
        "title": "LLM paper",
        "abstract_inverted_index": {"a": [0]},
    }

    class _StubModel:
        def invoke(self, messages, response_format=None):
            generation = ChatGeneration(message=AIMessage(content='{"article_type":"综述","citation_category":"high","score":88,"rationale":["覆盖面广"]}'))
            return ChatResult(generations=[generation])

    class _StubRouter:
        def create_chat_model(self, **kwargs):
            return _StubModel()

    assessor = LLMCitationAssessor(router=_StubRouter())  # type: ignore[arg-type]
    result = assessor.assess(work, heuristic={"category": "medium", "score": 2.0, "rationale": []})
    assert result["article_type"] == "综述"
    assert result["citation_category"] == "high"
    assert result["score"] == 88
