from __future__ import annotations

import pytest

from tiangong_ai_workspace.tooling.openalex import OpenAlexClient, OpenAlexClientError


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
