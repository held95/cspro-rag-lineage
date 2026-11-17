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
| Whole suite (main venv) | 168 passed, 6 skipped, 2 xfailed (the 9 `-m ollama` live tests run when Ollama is up, and skip otherwise) | `pytest -q` |
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

## Fine-tune (QLoRA, Qwen3-1.7B)

Datasets are synthetic and generated deterministically (seed 42): `finetune/data/{train,val,heldout,heldout_ood_handwritten}.jsonl`
(200 / 40 / 60 / 15 items; scenario families are disjoint between splits, `MANIFEST.json` has the SHA-256 of every file).

```
python -m finetune.gen_dataset --seed 42        # regenerate the data (optional; the files are committed)
python -m finetune.check_leakage                # train vs held-out overlap check (must print OK)

# training venv: see requirements-train.txt (torch cu128 + transformers 5, peft, trl, bitsandbytes), NVIDIA GPU
.venv-train\Scripts\python -m finetune.train --config finetune/configs/qlora_qwen3_1p7b.yaml --output-dir finetune/outputs/full
#   add --max-steps 20 for the smoke run

# one arm = generate + score; "base" has no adapter, "adapted" loads it. Repeat for heldout and ood.
.venv-train\Scripts\python -m finetune.generate --config finetune/configs/qlora_qwen3_1p7b.yaml --data finetune/data/heldout.jsonl --out finetune/outputs/preds/adapted__heldout.jsonl --adapter finetune/outputs/full/adapter
.venv-train\Scripts\python -m finetune.eval --preds finetune/outputs/preds/adapted__heldout.jsonl --gold finetune/data/heldout.jsonl --out finetune/results/adapted__heldout.json
python -m finetune.report --run-meta finetune/results/run_meta.json     # writes finetune/results/baseline_vs_adapted.md
```

For the baseline drop `--adapter` and name the files `base__heldout` / `base__ood`; the OOD set is
`finetune/data/heldout_ood_handwritten.jsonl` (files `*__ood`).

**Executed on 2026-11-17** (RTX 3050 6 GB): 75 steps, 1081 s, 2.2 GB peak VRAM; exact versions and settings are in
`finetune/results/run_meta.json`. Train loss ended near 0.0007 while validation loss stayed around 0.085-0.093, i.e. the
model memorised the templated training data; judge it by the held-out tables in `finetune/results/baseline_vs_adapted.md`,
not by the loss. Only two arms were run (base zero-shot and adapted); few-shot was not.

**Checkpoint.** The LoRA adapter (34.9 MB) is git-ignored (`finetune/outputs/`), so a fresh clone does not contain it: re-create it
with the training command above (about 18 minutes on the RTX 3050; the result is not guaranteed to be bit-identical).
The run reported here produced `adapter_model.safetensors` with SHA-256
`0e0f0afd2bcb998f0447b14d2147c8ff76c5cf9cc658647ec4b255e14cd3cbd3`. Base model: `Qwen/Qwen3-1.7B` at the commit pinned in the config.
To load it: `PeftModel.from_pretrained(base_model, "finetune/outputs/full/adapter")` (see `finetune/generate.py : load_model`).

Result summary (n=60 templated held-out, n=15 hand-written OOD, wide confidence intervals): the adapted model stops
over-abstaining (0.93 -> 0.00 held-out, 1.00 -> 0.09 OOD) and detects conflicts (F1 0 -> 0.77 / 0.75), while injection
compliance rose from 0 to 0.08 on the held-out set (CI [0, 0.27]) and unconstrained schema validity dipped slightly.
The adapter was **not** served through Ollama; the running app does not use it.

## Layout

```
app/api          routes, auth deps          app/modules/ingestion   ledger (source of truth), chunking, service
app/core         config, errors, middleware app/modules/retrieval   embedder, Qdrant store, retriever (+ledger re-check)
app/modules/tenancy   Principal, TenantScope, can_read   app/modules/generation   prompt, llm client, verifier, sanitiser
finetune         dataset, leakage, train, generate, eval, report     tests   unit / api / ingestion / security / finetune
```
