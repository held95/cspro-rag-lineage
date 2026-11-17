# Limitations and what was NOT done

## Not executed / not built
- **Live Ollama evidence is small**: 12-30 questions, one run, one seed (`docs/LIVE_RUN.md`). `Settings.min_score = 0.45` was
  checked, not tuned: answerable and unanswerable top-1 scores overlap (0.576 min vs 0.676 max), so no threshold separates them
  and the LLM's own abstention covers the rest. A first live run abstained on everything (schema without `required`); fixed.
- **Conflict detection with the real model is not reliable**: the P1 runbook/legacy conflict (KB-003 x KB-004) was flagged only
  when the question asked for conflicting information, and a ticket saying "14 days" was answered without flagging the
  current 30-day policy.
- **The Airflow DAG is not executed under a scheduler** (airflow is not installed). `LIN-04` is a static AST check and the stale
  retry is replayed through the HTTP API (`LIN-03`); `LIN-05` (DagBag in a container) was not done. No demo recording.
- **Not built, as the 24 h plan allows:** DELETE/tombstones, reconcile job, rate limiting, Hypothesis tests.
- **Security tests not written:** `SEC-16` (output canary tripwire; needs an app feature that does not exist), `SEC-17`
  (log injection; needs structured JSON logs) and `SEC-20` (real-model injection rate; only a handful of live questions touched the injection documents).
- **No test for the route-level 503 when the embedder or vector store is down on `/ask`** (the fakes do not simulate it).
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
- `/docs`, `/redoc` and `/openapi.json` are served without authentication (FastAPI defaults; informational). No rate limiting.
- API keys are static, hashed in config; no rotation, expiry or per-key rate limit.
- Qdrant runs in local mode (embedded); payload indexes and `is_tenant` indexing have no effect there.
- SQLite ledger: single node, one writer at a time (`BEGIN IMMEDIATE`).
