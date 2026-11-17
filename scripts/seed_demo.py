"""Seed the demo: random API keys -> .env.local, and ingest the synthetic records.

    python -m scripts.seed_demo [--embed hash|ollama] [--env-file .env.local] [--force]
                                [--ledger-path PATH] [--vector-path PATH]

Plaintext keys are printed ONCE; only their sha256 is stored in the env file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
from pathlib import Path

from app.container import build_container
from app.core.config import KeyRecord, Settings
from app.schemas.ingest import DocumentManifest
from app.modules.tenancy.principal import Principal

ROOT = Path(__file__).resolve().parent.parent
RECORDS = ROOT / "data" / "synthetic" / "records.json"

# Same subjects/groups/scopes as tests/conftest.py KEYS.
IDENTITIES = {
    "alpha-admin": KeyRecord(tenant_id="alpha", subject="alpha-admin", groups=["support", "oncall"],
                             scopes=["docs:read", "docs:write", "ask:read"]),
    "alpha-support": KeyRecord(tenant_id="alpha", subject="alpha-support", groups=["support"],
                               scopes=["docs:read", "ask:read"]),
    "beta-admin": KeyRecord(tenant_id="beta", subject="beta-admin", groups=[],
                            scopes=["docs:read", "docs:write", "ask:read"]),
}


def generate_keys() -> dict[str, str]:
    return {name: f"{name}-{secrets.token_urlsafe(24)}" for name in IDENTITIES}


def render_env(keys: dict[str, str], embed: str, ledger_path: str, vector_path: str) -> str:
    hashed = {hashlib.sha256(k.encode()).hexdigest(): IDENTITIES[n].model_dump() for n, k in keys.items()}
    lines = [
        f"CSPRO_EMBED_BACKEND={embed}",
        f"CSPRO_LEDGER_PATH={ledger_path}",
        f"CSPRO_VECTOR_PATH={vector_path}",
        # pydantic-settings parses dict fields from JSON; single quotes keep dotenv from touching it.
        "CSPRO_API_KEYS='" + json.dumps(hashed, separators=(",", ":")) + "'",
    ]
    return "\n".join(lines) + "\n"


def ingest_all(settings: Settings) -> int:
    container = build_container(settings)
    n = 0
    for r in json.loads(RECORDS.read_text("utf-8")):
        m = DocumentManifest(
            doc_id=r["doc_id"], version=r["version"], kind=r["kind"], title=r["title"], content=r["content"],
            topic=r.get("topic"), visibility=r["visibility"], acl_groups=r["acl_groups"],
            effective_date=r.get("effective_date"),
        )
        scope = Principal(tenant_id=r["tenant"], subject="seed", scopes=frozenset({"docs:write"})).scope()
        container.ingestion.ingest(scope, m)
        n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="scripts.seed_demo")
    p.add_argument("--embed", choices=["hash", "ollama"], default="ollama")
    p.add_argument("--env-file", default=".env.local")
    p.add_argument("--force", action="store_true", help="overwrite an existing env file")
    p.add_argument("--ledger-path", default="var/ledger.db")
    p.add_argument("--vector-path", default="var/qdrant")
    a = p.parse_args(argv)

    env_file = Path(a.env_file)
    if env_file.exists() and not a.force:
        print(f"{env_file} already exists; refusing to overwrite (use --force).", file=sys.stderr)
        return 1

    keys = generate_keys()
    for path in (Path(a.ledger_path).parent, Path(a.vector_path).parent):
        path.mkdir(parents=True, exist_ok=True)

    # Ingest FIRST, with settings built in memory: if this fails (e.g. the Qdrant folder is locked by a running
    # server), the existing env file is left untouched and no key is lost or invalidated.
    hashed = {hashlib.sha256(k.encode()).hexdigest(): IDENTITIES[n] for n, k in keys.items()}
    settings = Settings(_env_file=None, embed_backend=a.embed, ledger_path=a.ledger_path,
                        vector_path=a.vector_path, api_keys=hashed)
    try:
        count = ingest_all(settings)
    except RuntimeError as exc:
        if "already accessed by another instance" in str(exc):
            print(f"ERROR: {a.vector_path} is in use by another process (probably the running uvicorn server). "
                  "Stop the server (Ctrl+C in its window), run the seed again, then start the server. "
                  f"{env_file} was NOT changed.", file=sys.stderr)
            return 2
        raise

    env_file.write_text(render_env(keys, a.embed, a.ledger_path, a.vector_path), encoding="utf-8")
    print(f"Wrote {env_file}; ingested {count} records (embed={a.embed}).")
    print("API keys (shown ONCE, not stored in plaintext):")
    for name, key in keys.items():
        print(f"  {name}: {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
