# CSPRO RAG lineage demo

A small multi-tenant RAG service (FastAPI) that **reproduces and fixes** the "Airflow retry republishes a
superseded document" failure, enforces **tenant isolation in application code**, answers with
**source-verified citations** from a local LLM, and ships a **QLoRA fine-tune** with an honest
baseline-vs-adapted evaluation.

Read `docs/LIMITATIONS.md` and `docs/AI_DISCLOSURE.md` before judging the results.
`docs/WALKTHROUGH.md` maps every behaviour to a file, a test and a command; `docs/ARCHITECTURE.md` is the one-page design;
`docs/LIVE_RUN.md` and `docs/RETRIEVAL_EVAL.md` hold the (small) real-model measurements; `docs/SUBMISSION_CHECKLIST.md` the recording script.

## What is verified (and how)

| Claim | Evidence | Command |
|---|---|---|
| Naive pipeline resurrects a stale version, the fixed one does not | XFAIL (naive) / PASS (fixed) | `pytest -m failure_repro -v` |
| Tenant isolation, ACL, uniform 404 / abstention, prompt-injection and Unicode-smuggling containment | 27 tests | `pytest -m security -v` |
| Controls really matter: with them off the canary leaks | `SEC-24` tests | `pytest tests/security/test_injection.py -k sec24 -v` |
| Whole suite (main venv) | 158 passed, 6 skipped, 2 xfailed (the 9 `-m ollama` live tests run when Ollama is up, and skip otherwise) | `pytest -q` |
| QLoRA run on Qwen3-1.7B (RTX 3050, 6 GB) | `finetune/results/baseline_vs_adapted.md` (script-generated) | see "Fine-tune" |

The 6 skipped tests need the training venv (`transformers`, `pyyaml`) or a GPU (`-m gpu`).

## Setup

Python 3.12.

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt -r requirements-dev.txt
.venv\Scripts\python -m pytest -q
```

Tests use in-process fakes (hash embedder, in-memory Qdrant, scripted LLMs): no Ollama and no Docker needed.

## API

| Endpoint | Scope | Purpose |
|---|---|---|
| `GET /health` | none | liveness |
| `POST /ingest` | `docs:write` | ingest a document version (idempotent, stale-safe) |
| `GET /documents/{doc_id}` (+ lineage) | `docs:read` | document state; foreign / unauthorised ids are a plain 404 |
| `POST /ask` | `ask:read` | `{question, top_k}` -> `answered` / `abstained` / `conflict` with verified citations |

Auth: `X-API-Key`. The tenant, groups and scopes come **only** from the key record
(`Settings.api_keys`: `sha256(key) -> KeyRecord`); a tenant in the body or in a header is rejected or ignored.
Errors share one envelope: `{"error": {"code", "message", "details", "request_id"}}`.

## Running against real Ollama

The app defaults to Ollama (`qwen3:4b-instruct-2507-q4_K_M`, `nomic-embed-text`; configurable with `CSPRO_*`
environment variables, see `app/core/config.py`).

```
python -m scripts.seed_demo --embed ollama      # writes .env.local (only sha256 of the keys) and ingests the records; prints keys once
python -m uvicorn app.main:create_app --factory
python -m pytest -m ollama -v                   # live smoke (skips when Ollama is not reachable)
```

Executed on the author's machine (see `docs/LIVE_RUN.md`): a first live run abstained on every question because the JSON
Schema had no `required` keys; this was fixed and re-measured (9 answered / 3 correctly abstained of 12). Samples are tiny
(12-30 questions, one run). Not reliable yet: conflict detection (only flagged when the question invites it) and a ticket
that contradicts the current policy.

## Models and licences
LLM `qwen3:4b-instruct-2507-q4_K_M` (fallback `qwen3:1.7b-q4_K_M`, also the fine-tune base `Qwen/Qwen3-1.7B`) and embedder
`nomic-embed-text` run locally through Ollama. The research notes (`docs/research/`) list them as Apache-2.0, but that was
**not re-verified against the model cards**; check the licences before any reuse. Fine-tune pins: `requirements-train.txt`.

## Fine-tune

Pipeline: `finetune.gen_dataset` -> `finetune.check_leakage` -> `finetune.train` -> `finetune.generate` +
`finetune.eval` per arm -> `finetune.report`. Training needs the separate train venv (`.venv-train`: torch cu128,
transformers 5, peft, trl, bitsandbytes) and a free GPU.

```
.venv-train\Scripts\python -m finetune.train --config finetune/configs/qlora_qwen3_1p7b.yaml --output-dir finetune/outputs/full
```

Executed: 75 steps, 1081 s, 2.2 GB peak VRAM. Train loss ended near 0.0007 while validation loss stayed
around 0.085-0.093, i.e. the model memorised the templated training data; judge it by the held-out tables, not by the
loss. The adapter itself is git-ignored (`finetune/outputs/`).

Result summary (n=60 templated held-out, n=15 hand-written OOD, wide confidence intervals): the adapted model stops
over-abstaining (0.93 -> 0.00 held-out, 1.00 -> 0.09 OOD) and detects conflicts (F1 0 -> 0.77 / 0.75), while injection
compliance rose from 0 to 0.08 on the held-out set (CI [0, 0.27]) and unconstrained schema validity dipped slightly.
The adapter was **not** served through Ollama; the app never uses it.

## Layout

```
app/api          routes, auth deps          app/modules/ingestion   ledger (source of truth), chunking, service
app/core         config, errors, middleware app/modules/retrieval   embedder, Qdrant store, retriever (+ledger re-check)
app/modules/tenancy   Principal, TenantScope, can_read   app/modules/generation   prompt, llm client, verifier, sanitiser
finetune         dataset, leakage, train, generate, eval, report     tests   unit / api / ingestion / security / finetune
```
