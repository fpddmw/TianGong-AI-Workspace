from __future__ import annotations

import json

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


def test_llm_assessor_parses_json(monkeypatch: pytest.MonkeyPatch) -> None:
    work = {
        "title": "LLM paper",
        "abstract_inverted_index": {"a": [0]},
    }

    class _StubModel:
        def invoke(self, messages, response_format=None):
            payload = {
                "prediction": {
                    "estimated_band": "High",
                    "key_reason": "覆盖面广，方法严谨。",
                },
                "dimension_scores": {
                    "topic": {"score": 3, "eval": "优", "analysis": "跨学科视角"},
                    "methodology": {"score": 3, "eval": "优", "analysis": "方法闭环完整"},
                    "data": {"score": 2, "eval": "中", "analysis": "数据量一般"},
                    "impact": {"score": 3, "eval": "优", "analysis": "决策指向明确"},
                },
                "action_plan": ["补充更大规模的数据集"],
            }
            generation = ChatGeneration(message=AIMessage(content=json.dumps(payload, ensure_ascii=False)))
            return ChatResult(generations=[generation])

    class _StubRouter:
        def create_chat_model(self, **kwargs):
            return _StubModel()

    assessor = LLMCitationAssessor(router=_StubRouter())  # type: ignore[arg-type]
    result = assessor.assess(work)
    assert result["prediction"]["estimated_band"] == "High"
    assert result["dimension_scores"]["topic"]["score"] == 3
    assert result["action_plan"][0].startswith("补充")
