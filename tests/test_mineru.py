from __future__ import annotations

import pytest

from tiangong_ai_workspace.secrets import MineruSecrets, Secrets
from tiangong_ai_workspace.tooling.mineru import MineruClient, MineruClientError


def test_mineru_client_missing_config_raises() -> None:
    secrets = Secrets(openai=None, mcp_servers={}, mineru=None)
    with pytest.raises(MineruClientError):
        MineruClient(secrets=secrets)


def test_mineru_client_sends_payload(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    config = MineruSecrets(api_url="https://example.com/mineru_with_images", token="abc123")
    secrets = Secrets(openai=None, mcp_servers={}, mineru=config)
    client = MineruClient(secrets=secrets, timeout=1.0)

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"pdf-bytes")

    captured: dict[str, object] = {}

    class _StubResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"ok": True}

    def fake_post(self, url: str, *, files, data, headers):
        captured["url"] = url
        captured["data"] = dict(data)
        captured["headers"] = dict(headers)
        captured["file_tuple"] = files["file"]
        return _StubResponse()

    monkeypatch.setattr(MineruClient, "_post", fake_post, raising=False)

    result = client.recognize_with_images(
        pdf_path,
        prompt="check images",
        save_to_minio=True,
        minio_address="http://minio",
        minio_access_key="ak",
        minio_secret_key="sk",
        minio_bucket="bucket",
        minio_prefix="prefix/",
        minio_meta="meta",
        provider="openai",
        model="gpt",
    )

    assert captured["url"] == "https://example.com/mineru_with_images"
    assert captured["headers"]["Authorization"] == "Bearer abc123"
    assert captured["data"]["prompt"] == "check images"
    assert captured["data"]["save_to_minio"] == "true"
    assert captured["data"]["minio_bucket"] == "bucket"
    assert captured["data"]["provider"] == "openai"
    assert captured["file_tuple"][0] == "sample.pdf"
    assert result["result"]["ok"] is True
