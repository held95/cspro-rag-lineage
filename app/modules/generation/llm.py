from __future__ import annotations

from typing import Protocol

import httpx


class LLMClient(Protocol):
    name: str

    def generate(self, messages: list[dict[str, str]], schema: dict | None = None) -> str: ...


class OllamaClient:
    """Local open-weight model through Ollama. Output is constrained to a JSON Schema (``format``)."""

    def __init__(self, base_url: str, model: str, connect_timeout: float = 5.0, read_timeout: float = 120.0):
        self.base_url, self.name = base_url.rstrip("/"), model
        self._client = httpx.Client(timeout=httpx.Timeout(read_timeout, connect=connect_timeout))

    def generate(self, messages: list[dict[str, str]], schema: dict | None = None) -> str:
        payload = {
            "model": self.name, "messages": messages, "stream": False,
            "options": {"temperature": 0, "seed": 42, "num_ctx": 4096},
        }
        if schema is not None:
            payload["format"] = schema
        r = self._client.post(f"{self.base_url}/api/chat", json=payload)
        r.raise_for_status()
        return r.json()["message"]["content"]
