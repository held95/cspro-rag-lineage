"""SQLite ledger = source of truth for document versions and lineage.

Two counters per document keep stale replays and serving apart:
  * latest_accepted_version - the stale gate: anything older than this is rejected
  * active_version          - what retrieval is allowed to serve

A version is accepted as ``pending`` first; it only becomes ``active`` (and supersedes the previous
one) after its vectors were written. A crash or outage in between therefore never leaves the
document dark, and an Airflow retry can never move the document backwards.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

from app.schemas.ingest import DocumentManifest, RunContext


class IngestOutcome(StrEnum):
    ACCEPTED = "accepted"
    NOOP_DUPLICATE = "noop_duplicate"
    REJECTED_STALE = "rejected_stale"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class Decision:
    outcome: IngestOutcome
    needs_vectors: bool


@dataclass(frozen=True)
class VersionRow:
    tenant_id: str
    doc_id: str
    version: int
    status: str
    content: str
    content_sha256: str
    kind: str
    title: str
    topic: str | None
    visibility: str
    acl_groups: list[str]
    source_uri: str | None
    effective_date: str | None
    ingested_at: str
    activated_at: str | None
    run_id: str | None
    try_number: int | None


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents(
  tenant_id TEXT NOT NULL, doc_id TEXT NOT NULL,
  latest_accepted_version INTEGER NOT NULL, active_version INTEGER, updated_at TEXT NOT NULL,
  PRIMARY KEY (tenant_id, doc_id));
CREATE TABLE IF NOT EXISTS document_versions(
  tenant_id TEXT NOT NULL, doc_id TEXT NOT NULL, version INTEGER NOT NULL,
  content TEXT NOT NULL, content_sha256 TEXT NOT NULL, kind TEXT NOT NULL, status TEXT NOT NULL,
  idempotency_key TEXT NOT NULL UNIQUE, title TEXT NOT NULL, topic TEXT, visibility TEXT NOT NULL,
  acl_groups_json TEXT NOT NULL, source_uri TEXT, effective_date TEXT,
  ingested_at TEXT NOT NULL, activated_at TEXT, embedding_model TEXT,
  run_id TEXT, dag_id TEXT, task_id TEXT, try_number INTEGER, data_interval_start TEXT,
  PRIMARY KEY (tenant_id, doc_id, version));
CREATE TABLE IF NOT EXISTS ingestion_events(
  event_id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
  tenant_id TEXT NOT NULL, doc_id TEXT NOT NULL, version INTEGER,
  action TEXT NOT NULL, reason TEXT, run_id TEXT, try_number INTEGER, idempotency_key TEXT);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def idempotency_key(tenant_id: str, doc_id: str, version: int, content_sha256: str) -> str:
    return hashlib.sha256(f"{tenant_id}|{doc_id}|{version}|{content_sha256}".encode()).hexdigest()


class Ledger:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._local = threading.local()  # one connection per thread
        self._conn().executescript(SCHEMA)  # manages its own transaction

    # ---- connection / transactions ----------------------------------------------------------
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, isolation_level=None, timeout=5.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return conn

    def _tx(self):
        return _Tx(self._conn())

    # ---- step 1: accept (stale gate, idempotency, content conflict) --------------------------
    def accept(self, tenant_id: str, m: DocumentManifest, run: RunContext | None, embed_model: str) -> Decision:
        sha = hashlib.sha256(m.content.encode()).hexdigest()
        key = idempotency_key(tenant_id, m.doc_id, m.version, sha)
        run = run or RunContext()
        with self._tx() as c:  # BEGIN IMMEDIATE: one writer at a time
            row = c.execute(
                "SELECT latest_accepted_version FROM documents WHERE tenant_id=? AND doc_id=?",
                (tenant_id, m.doc_id),
            ).fetchone()
            if row is not None:
                latest = row["latest_accepted_version"]
                # 1) STALE FIRST. An Airflow retry replays the identical payload (same idempotency
                #    key), so the order check must come before the duplicate check.
                if m.version < latest:
                    self._event(c, tenant_id, m.doc_id, m.version, "rejected_stale",
                                f"latest_accepted_version={latest}", run, key)
                    return Decision(IngestOutcome.REJECTED_STALE, False)
                if m.version == latest:
                    existing = c.execute(
                        "SELECT content_sha256, status FROM document_versions WHERE tenant_id=? AND doc_id=? AND version=?",
                        (tenant_id, m.doc_id, m.version),
                    ).fetchone()
                    if existing["content_sha256"] != sha:
                        self._event(c, tenant_id, m.doc_id, m.version, "conflict",
                                    "same version, different content", run, key)
                        return Decision(IngestOutcome.CONFLICT, False)
                    self._event(c, tenant_id, m.doc_id, m.version, "noop_duplicate", None, run, key)
                    return Decision(IngestOutcome.NOOP_DUPLICATE, existing["status"] == "pending")
                changed = c.execute(
                    "UPDATE documents SET latest_accepted_version=?, updated_at=? "
                    "WHERE tenant_id=? AND doc_id=? AND latest_accepted_version<?",
                    (m.version, _now(), tenant_id, m.doc_id, m.version),
                ).rowcount
                assert changed == 1
            else:
                c.execute(
                    "INSERT INTO documents(tenant_id, doc_id, latest_accepted_version, active_version, updated_at) "
                    "VALUES (?,?,?,NULL,?)", (tenant_id, m.doc_id, m.version, _now()),
                )
            c.execute(
                "INSERT INTO document_versions(tenant_id, doc_id, version, content, content_sha256, kind, status, "
                "idempotency_key, title, topic, visibility, acl_groups_json, source_uri, effective_date, ingested_at, "
                "embedding_model, run_id, dag_id, task_id, try_number, data_interval_start) "
                "VALUES (?,?,?,?,?,?, 'pending', ?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (tenant_id, m.doc_id, m.version, m.content, sha, m.kind, key, m.title, m.topic, m.visibility,
                 json.dumps(m.acl_groups), m.source_uri, m.effective_date.isoformat() if m.effective_date else None,
                 _now(), embed_model, run.run_id, run.dag_id, run.task_id, run.try_number, run.data_interval_start),
            )
            self._event(c, tenant_id, m.doc_id, m.version, "accepted", None, run, key)
            return Decision(IngestOutcome.ACCEPTED, True)

    # ---- step 2: activate (supersede the previous version) ------------------------------------
    def activate(self, tenant_id: str, doc_id: str, version: int) -> bool:
        """Make ``version`` the served version iff it is still the newest accepted one."""
        with self._tx() as c:
            won = c.execute(
                "UPDATE documents SET active_version=?, updated_at=? WHERE tenant_id=? AND doc_id=? "
                "AND latest_accepted_version=? AND (active_version IS NULL OR active_version<?)",
                (version, _now(), tenant_id, doc_id, version, version),
            ).rowcount == 1
            if not won:
                # a newer version already won (or this one is not the newest): never serve it
                c.execute("UPDATE document_versions SET status='superseded' WHERE tenant_id=? AND doc_id=? "
                          "AND version=? AND status='pending'", (tenant_id, doc_id, version))
                self._event(c, tenant_id, doc_id, version, "superseded", "newer version won", RunContext(), None)
                return False
            c.execute("UPDATE document_versions SET status='superseded' WHERE tenant_id=? AND doc_id=? "
                      "AND version<? AND status IN ('active','pending')", (tenant_id, doc_id, version))
            c.execute("UPDATE document_versions SET status='active', activated_at=? WHERE tenant_id=? AND doc_id=? "
                      "AND version=?", (_now(), tenant_id, doc_id, version))
            self._event(c, tenant_id, doc_id, version, "activated", None, RunContext(), None)
            return True

    # ---- reads --------------------------------------------------------------------------------
    def get_active_version(self, tenant_id: str, doc_id: str) -> int | None:
        row = self._conn().execute(
            "SELECT active_version FROM documents WHERE tenant_id=? AND doc_id=?", (tenant_id, doc_id)
        ).fetchone()
        return row["active_version"] if row else None

    def get_version(self, tenant_id: str, doc_id: str, version: int) -> VersionRow | None:
        row = self._conn().execute(
            "SELECT * FROM document_versions WHERE tenant_id=? AND doc_id=? AND version=?",
            (tenant_id, doc_id, version),
        ).fetchone()
        return self._row(row) if row else None

    def versions(self, tenant_id: str, doc_id: str) -> list[VersionRow]:
        rows = self._conn().execute(
            "SELECT * FROM document_versions WHERE tenant_id=? AND doc_id=? ORDER BY version", (tenant_id, doc_id)
        ).fetchall()
        return [self._row(r) for r in rows]

    def events(self, tenant_id: str, doc_id: str) -> list[dict]:
        rows = self._conn().execute(
            "SELECT ts, version, action, reason, run_id, try_number FROM ingestion_events "
            "WHERE tenant_id=? AND doc_id=? ORDER BY event_id", (tenant_id, doc_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def record_event(self, tenant_id: str, doc_id: str, version: int | None, action: str, reason: str | None) -> None:
        with self._tx() as c:
            self._event(c, tenant_id, doc_id, version, action, reason, RunContext(), None)

    # ---- helpers ------------------------------------------------------------------------------
    @staticmethod
    def _event(c, tenant_id, doc_id, version, action, reason, run: RunContext, key):
        c.execute(
            "INSERT INTO ingestion_events(ts, tenant_id, doc_id, version, action, reason, run_id, try_number, idempotency_key) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (_now(), tenant_id, doc_id, version, action, reason, run.run_id, run.try_number, key),
        )

    @staticmethod
    def _row(r: sqlite3.Row) -> VersionRow:
        return VersionRow(
            r["tenant_id"], r["doc_id"], r["version"], r["status"], r["content"], r["content_sha256"], r["kind"],
            r["title"], r["topic"], r["visibility"], json.loads(r["acl_groups_json"]), r["source_uri"],
            r["effective_date"], r["ingested_at"], r["activated_at"], r["run_id"], r["try_number"],
        )


class _Tx:
    """`with ledger._tx() as conn:` -> BEGIN IMMEDIATE ... COMMIT/ROLLBACK."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def __enter__(self) -> sqlite3.Connection:
        self.conn.execute("BEGIN IMMEDIATE")
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        self.conn.execute("ROLLBACK" if exc_type else "COMMIT")
        return False
