from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import httpx


class Embedder(Protocol):
    dim: int
    name: str

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]: ...


class HashEmbedder:
    """Deterministic, download-free embedder for tests: signed feature hashing of lower-cased word
    unigrams + bigrams into ``dim`` buckets, L2-normalised. Overlapping words => high cosine.
    Its retrieval numbers are PIPELINE tests, not a claim about semantic quality."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hash-{dim}"

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        feats = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        vec = [0.0] * self.dim
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


class OllamaEmbedder:
    """nomic-embed-text via Ollama /api/embed (needs the task prefixes)."""

    def __init__(self, base_url: str, model: str = "nomic-embed-text", dim: int = 768, timeout: float = 60.0):
        self.base_url, self.model, self.dim, self.name = base_url.rstrip("/"), model, dim, model
        self._client = httpx.Client(timeout=timeout)

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]:
        prefix = "search_query: " if kind == "query" else "search_document: "
        r = self._client.post(f"{self.base_url}/api/embed", json={"model": self.model, "input": [prefix + t for t in texts]})
        r.raise_for_status()
        return r.json()["embeddings"]
