"""scripts.seed_demo with --embed hash, entirely under tmp_path (never touches the repo's var/ or .env.local)."""
from __future__ import annotations

import hashlib
import re
import sqlite3

import pytest

from app.core.config import Settings
from scripts import seed_demo

pytestmark = pytest.mark.unit


def _run(tmp_path, *extra):
    env = tmp_path / ".env.test"
    argv = ["--embed", "hash", "--env-file", str(env), "--ledger-path", str(tmp_path / "ledger.db"),
            "--vector-path", str(tmp_path / "qdrant"), *extra]
    return env, argv


def test_seed_writes_env_loads_settings_and_ingests_all(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)  # any stray relative path lands in tmp, not in the repo
    env, argv = _run(tmp_path)
    assert seed_demo.main(argv) == 0
    out = capsys.readouterr().out

    settings = Settings(_env_file=str(env))
    assert settings.embed_backend == "hash"
    assert len(settings.api_keys) == 3
    printed = dict(re.findall(r"^\s+(alpha-admin|alpha-support|beta-admin): (\S+)$", out, re.M))
    assert set(printed) == {"alpha-admin", "alpha-support", "beta-admin"}
    for name, key in printed.items():
        assert settings.api_keys[hashlib.sha256(key.encode()).hexdigest()].subject == name
        assert key not in env.read_text("utf-8")  # plaintext never stored
    assert "docs:write" not in settings.api_keys[hashlib.sha256(printed["alpha-support"].encode()).hexdigest()].scopes

    n = sqlite3.connect(tmp_path / "ledger.db").execute("SELECT COUNT(*) FROM document_versions").fetchone()[0]
    assert n == len(seed_demo.json.loads(seed_demo.RECORDS.read_text("utf-8")))
    assert not (tmp_path / "var").exists()


def test_seed_refuses_to_overwrite_without_force(tmp_path, capsys):
    env, argv = _run(tmp_path)
    env.write_text("KEEP=me\n", encoding="utf-8")
    assert seed_demo.main(argv) == 1
    assert env.read_text("utf-8") == "KEEP=me\n"
    assert seed_demo.main([*argv, "--force"]) == 0
    assert "CSPRO_API_KEYS" in env.read_text("utf-8")


def test_failed_ingestion_leaves_env_file_untouched_and_prints_no_keys(tmp_path, capsys, monkeypatch):
    """Regression: the seed used to overwrite the env file BEFORE ingesting. A locked Qdrant folder (server running)
    then left new, never-printed key hashes behind and invalidated the keys the user already had."""
    env, argv = _run(tmp_path)
    env.write_text("KEEP=me\n", encoding="utf-8")

    def locked(_settings):
        raise RuntimeError("Storage folder var/qdrant is already accessed by another instance of Qdrant client.")

    monkeypatch.setattr(seed_demo, "ingest_all", locked)
    assert seed_demo.main([*argv, "--force"]) == 2
    captured = capsys.readouterr()
    assert env.read_text("utf-8") == "KEEP=me\n"
    assert "Stop the server" in captured.err
    assert "alpha-admin" not in captured.out
