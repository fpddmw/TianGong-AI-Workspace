"""
Structured response helpers for agent-friendly CLI output.

The :class:`WorkspaceResponse` dataclass ensures that every CLI command can emit
predictable JSON while remaining human-readable by default. Agents can inspect
the `status` and `errors` fields to decide whether to retry or branch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, Mapping, MutableMapping, TypeVar

__all__ = ["ResponsePayload", "WorkspaceResponse"]

ResponsePayload = TypeVar("ResponsePayload")


@dataclass(slots=True, frozen=True)
class WorkspaceResponse(Generic[ResponsePayload]):
    """Structured response envelope shared across CLI commands."""

    status: Literal["success", "warning", "error"]
    message: str
    payload: ResponsePayload | None = None
    errors: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> MutableMapping[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        data: MutableMapping[str, Any] = {
            "status": self.status,
            "message": self.message,
        }
        if self.payload is not None:
            data["payload"] = self.payload
        if self.errors:
            data["errors"] = list(self.errors)
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise the response using `json.dumps`."""
        # Use ensure_ascii=False so that non-ASCII (e.g. Chinese) characters
        # are preserved as UTF-8 characters in the JSON output rather than
        # escaped \u... sequences. The CLI writes the JSON bytes as UTF-8
        # to stdout when requested, ensuring correct display when
        # redirecting output to files.
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @staticmethod
    def ok(payload: ResponsePayload | None = None, message: str = "OK", **metadata: Any) -> "WorkspaceResponse[ResponsePayload]":
        """Convenience constructor for nominal responses."""
        meta = metadata or None
        return WorkspaceResponse(status="success", message=message, payload=payload, metadata=meta)

    @staticmethod
    def warn(message: str, payload: ResponsePayload | None = None, *, errors: tuple[str, ...] | None = None, **metadata: Any) -> "WorkspaceResponse[ResponsePayload]":
        """Convenience constructor for warning responses."""
        meta = metadata or None
        errs = errors or tuple()
        return WorkspaceResponse(status="warning", message=message, payload=payload, errors=errs, metadata=meta)

    @staticmethod
    def error(message: str, *, errors: tuple[str, ...] | None = None, **metadata: Any) -> "WorkspaceResponse[None]":
        """Convenience constructor for error responses."""
        meta = metadata or None
        errs = errors or tuple()
        return WorkspaceResponse(status="error", message=message, payload=None, errors=errs, metadata=meta)
