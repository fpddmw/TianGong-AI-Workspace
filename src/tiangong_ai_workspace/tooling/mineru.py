"""Mineru PDF image extraction client."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

import httpx

from ..secrets import MineruSecrets, Secrets, load_secrets

__all__ = ["MineruClient", "MineruClientError"]


class MineruClientError(RuntimeError):
    """Raised when the Mineru API request fails."""


@dataclass(slots=True)
class MineruClient:
    """Lightweight wrapper for the Mineru `/mineru_with_images` endpoint."""

    secrets: Optional[Secrets] = None
    api_url_override: Optional[str] = None
    token_override: Optional[str] = None
    timeout: float = 60.0
    http_client: Optional[httpx.Client] = None
    _config: MineruSecrets = field(init=False, repr=False)

    def __post_init__(self) -> None:
        loaded = self.secrets or load_secrets()
        config = loaded.mineru
        api_url = self.api_url_override or (config.api_url if config else None)
        token = self.token_override or (config.token if config else None)
        if not api_url or not token:
            raise MineruClientError("Mineru API configuration is missing. Set [mineru] in secrets.toml or pass --token/--url.")

        normalized = MineruSecrets(api_url=api_url.rstrip("/"), token=token)
        object.__setattr__(self, "secrets", loaded)
        object.__setattr__(self, "_config", normalized)

    def recognize_with_images(
        self,
        file_path: Path,
        *,
        prompt: Optional[str] = None,
        save_to_minio: bool = False,
        minio_address: Optional[str] = None,
        minio_access_key: Optional[str] = None,
        minio_secret_key: Optional[str] = None,
        minio_bucket: Optional[str] = None,
        minio_prefix: Optional[str] = None,
        minio_meta: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Mapping[str, Any]:
        """Send a PDF to the Mineru API and return the parsed JSON response."""

        if not file_path.exists() or not file_path.is_file():
            raise MineruClientError(f"File not found: {file_path}")

        form_data: MutableMapping[str, Any] = {}
        if prompt is not None:
            form_data["prompt"] = prompt
        form_data["save_to_minio"] = "true" if save_to_minio else "false"
        if minio_address:
            form_data["minio_address"] = minio_address
        if minio_access_key:
            form_data["minio_access_key"] = minio_access_key
        if minio_secret_key:
            form_data["minio_secret_key"] = minio_secret_key
        if minio_bucket:
            form_data["minio_bucket"] = minio_bucket
        if minio_prefix:
            form_data["minio_prefix"] = minio_prefix
        if minio_meta:
            form_data["minio_meta"] = minio_meta
        if provider:
            form_data["provider"] = provider
        if model:
            form_data["model"] = model

        url = self._config.api_url
        headers = {"Authorization": f"Bearer {self._config.token}"}

        with file_path.open("rb") as handle:
            files = {"file": (file_path.name, handle, "application/pdf")}
            try:
                response = self._post(url, files=files, data=form_data, headers=headers)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise MineruClientError(f"Mineru API request failed: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:  # pragma: no cover - defensive fallback
            raise MineruClientError("Mineru API returned invalid JSON.") from exc

        return {
            "request": {"endpoint": url, "file": file_path.name},
            "result": data,
        }

    def _post(self, url: str, *, files: Mapping[str, Any], data: Mapping[str, Any], headers: Mapping[str, str]) -> httpx.Response:
        if self.http_client is not None:
            return self.http_client.post(url, files=files, data=data, headers=headers, timeout=self.timeout)
        return httpx.post(url, files=files, data=data, headers=headers, timeout=self.timeout)
