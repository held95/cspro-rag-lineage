# Limitations and what was NOT done

## Not executed / not built
- **No live Ollama run.** `/ask` was only exercised with scripted fake LLMs and a hash embedder. The live smoke
  (`LLM-02`), the retrieval threshold calibration on real embeddings (`Settings.min_score = 0.45` is an untuned default;
  tests use 0.05 with the hash embedder) and any real-model quality claim are missing.
- **No Airflow DAG** (`dags/ingest_dag.py` does not exist) and no tests for it (`LIN-03/04/05`). The lineage fix is shown
  by the in-process naive-vs-fixed test (`LIN-01/02`) and the ingestion tests, not by a scheduler.
- **No `scripts/seed_demo.py`** (`config.py` still mentions it) and no demo recording.
- **Not built, as the 24 h plan allows:** DELETE/tombstones, reconcile job, rate limiting, Hypothesis tests.
- **Security tests not written:** `SEC-16` (output canary tripwire; needs an app feature that does not exist), `SEC-17`
  (log injection; needs structured JSON logs) and `SEC-20` (real-model injection rate; needs Ollama).
- **`tests/generation/` is empty.** The verifier and `AskService` have no dedicated tests (malformed-JSON retry, 503/504 mapping,
  wrong-number answers, conflict detection): the fake LLMs exist in `tests/fakes.py` but nothing uses them yet.
- **Adapter not served.** The LoRA adapter was evaluated with Hugging Face generation only; merging/GGUF/Ollama serving
  was not attempted, so the running app does not use the fine-tuned model.

## Known weaknesses of what was measured
- **Fine-tune data are synthetic and templated.** Held-out items come from unseen scenario *families* (leakage check:
  mean 8-gram overlap 0.073, threshold 0.3) but share the generator's style. Train loss ~0.0007 against flat validation
  loss shows memorisation. The 15 hand-written OOD items are the only non-templated evidence, and n=15 gives intervals such
  as conflict recall 1.0 [0, 1].
- **Regressions of the adapted model:** injection compliance 0 -> 0.083 on held-out (CI [0, 0.27]); unconstrained schema
  validity 0.983 -> 0.967 (held-out) and 1.0 -> 0.933 (OOD). Conflict precision is only 0.63 / 0.60 (false alarms).
  The served path constrains output with a JSON schema, so the schema row does not transfer.
- **Same prompt for both arms**, and training used the exact evaluation prompt format; gains may not transfer to other prompts.
- **Single seed, one run.** No variance across seeds.

## Security scope
- Isolation is tested against scripted adversaries ("obedient" fake LLM), not against a real model's behaviour.
- `SEC-24` removes defences with test-only wrappers to show the tests can fail; it demonstrates the tests, not production hardening.
- API keys are static, hashed in config; no rotation, expiry or per-key rate limit.
- Qdrant runs in local mode (embedded); payload indexes and `is_tenant` indexing have no effect there.
- SQLite ledger: single node, one writer at a time (`BEGIN IMMEDIATE`).
