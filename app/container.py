"""Wires the real components from Settings. Tests build a Container by hand with fakes."""
from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings
from app.modules.generation.llm import OllamaClient
from app.modules.generation.service import AskService
from app.modules.ingestion.ledger import Ledger
from app.modules.ingestion.service import IngestionService
from app.modules.retrieval.embedder import HashEmbedder, OllamaEmbedder
from app.modules.retrieval.qdrant_store import QdrantStore
from app.modules.retrieval.retriever import Retriever


@dataclass
class Container:
    ledger: Ledger
    store: object
    embedder: object
    ingestion: IngestionService
    retriever: Retriever
    ask: AskService
    llm: object
    embed_model: str


def build_container(settings: Settings) -> Container:
    embedder = HashEmbedder() if settings.embed_backend == "hash" else OllamaEmbedder(settings.ollama_url, settings.embed_model)
    store = QdrantStore(path=settings.vector_path, collection=settings.collection, dim=embedder.dim)
    ledger = Ledger(settings.ledger_path)
    if settings.llm_backend == "ollama":
        llm = OllamaClient(settings.ollama_url, settings.llm_model, settings.llm_connect_timeout, settings.llm_read_timeout)
    else:
        raise ValueError("llm_backend=fake is only available through tests (inject a Container)")
    retriever = Retriever(ledger, store, embedder, settings.min_score)
    return Container(ledger, store, embedder, IngestionService(ledger, store, embedder, embedder.name),
                     retriever, AskService(retriever, llm, llm.name), llm, embedder.name)
