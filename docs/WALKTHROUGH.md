# Walkthrough: behaviour -> file -> test -> command

Run a single test: `python -m pytest -x -q -k <test_name>`; a group: `-m security`, `-m failure_repro`.

| Behaviour | File : function | Test | Command (`-k`) |
|---|---|---|---|
| Stale version is rejected BEFORE the idempotency check | `app/modules/ingestion/ledger.py : Ledger.accept` | `test_lin01_...[naive/fixed]`, `test_ing04...`, `test_lin03_...` | `lin01`, `ing04`, `lin03` |
| Replayed retry (same payload) is 409, not a republish | `ledger.py : accept` + `app/api/routes/ingest.py` | `test_lin03_replayed_stale_version_is_409_...` | `lin03` |
| Same version, different content is a conflict | `ledger.py : accept` | `test_ing06...`, `test_same_version_different_content_is_409_conflict` | `ing06` |
| Any arrival order ends with the highest version active | `ledger.py : accept/activate` | `test_ing05...` | `ing05` |
| Crash/embedder failure never serves a half-written version | `app/modules/ingestion/service.py : ingest` | `test_ing08`, `test_ing09`, `test_ing10` | `ing08 or ing09 or ing10` |
| Same `doc_id` in two tenants stays independent | `ledger.py` (keyed by tenant) | `test_ing12...`, `test_sec21...` | `ing12 or sec21` |
| Tenant comes only from the API key | `app/api/deps.py`, `app/modules/tenancy/principal.py` | `test_sec03...`, `test_sec04...` | `sec03 or sec04` |
| ANN query is tenant+ACL filtered; store cannot be used without a `TenantScope` | `app/modules/retrieval/qdrant_store.py : search` | `test_sec22...`, `test_sec21...` | `sec22` |
| Independent ledger re-check on every hit | `app/modules/retrieval/retriever.py : _authorized` | `test_sec05...`, `test_sec24_ledger_recheck_alone...` | `sec05 or sec24` |
| Quote must appear verbatim in the cited source; unknown aliases dropped | `app/modules/generation/verify.py : verify_answer`, `normalize.py : find_quote` | `test_sec06...`, `tests/unit/test_normalize.py` | `sec06`, `-k "quote"` |
| Prompt: random boundary, boundary lookalikes neutralised | `app/modules/generation/prompt.py : build_messages/_neutralise` | `test_sec13...` | `sec13` |
| Output: images, links, URLs stripped from answer and excerpts | `app/modules/generation/sanitize.py : sanitize` | `test_sec14...`, `test_sec25...` | `sec14 or sec25` |
| Obedient LLM cannot leak beta's canary | whole `/ask` path | `test_sec01...` | `sec01` |
| Same attack leaks when the controls are off | `tests/security/controls_off.py` | `test_sec24_all_controls_off...` | `sec24` |
| 404 uniformity, abstention uniformity | `app/api/routes/documents.py`, `service.py` | `test_sec07`, `test_sec08`, `test_sec23` | `sec07 or sec08 or sec23` |
| Error envelope, no input echo, no internals | `app/core/errors.py` | `test_api05_06...`, `test_api10...`, `test_dependency_outage...` | `api05 or api10 or outage` |
| Train prompt == serve prompt | `finetune/render.py`, `app/modules/generation/prompt.py` | `test_ft10...` | `ft10` |
| Dataset leakage check | `finetune/check_leakage.py` | `test_ft02...`, `test_ft03...` | `ft02 or ft03` |

## The stale-retry fix in ten lines
1. Airflow retries re-send the identical payload of an older run, after a newer version was already accepted.
2. A naive pipeline treats "same key seen?" as the only guard, then re-activates the old version.
3. The ledger keeps `latest_accepted_version` separately from `active_version`.
4. `Ledger.accept` checks `version < latest` FIRST and returns `REJECTED_STALE` (HTTP 409).
5. Only then does it look at equal versions: same hash is a duplicate (200), different hash a conflict (409).
6. New versions are written as `pending`, vectors are written, then `activate` flips them in one transaction.
7. `activate` succeeds only if the version is still the newest; otherwise its own points are deleted.
8. Retrieval re-checks every hit against the ledger, so a leftover point of an old version is never served.
9. The naive pipeline lives in `tests/naive_pipeline.py` and fails `LIN-01/02` (XFAIL); the real one passes.
10. A DAG that calls `/ingest` was NOT built; the retry is replayed through the HTTP API in `test_lin03`.

## Security layers in ten lines
1. Tenant, groups and scopes come only from the key record. 2. `TenantScope` can only be minted from a `Principal`.
3. Every store call needs a scope. 4. The ANN filter includes tenant and ACL. 5. The ledger re-checks tenant, status,
active version and ACL on each hit. 6. The model sees only aliases `S1..Sn` and a random per-request boundary.
7. Retrieved text is neutralised and framed as untrusted. 8. The verifier accepts only this request's aliases and
verbatim quotes. 9. Numbers in the answer must come from a verified quote. 10. The sanitiser strips images, links and URLs.

## Not covered by dedicated tests (be upfront)
- `tests/generation/` is empty: `verify_answer` and `AskService` are covered only through the security suite. Malformed
  JSON retry, timeouts (504), outages (503), wrong-number answers and conflict detection have fake LLMs in `tests/fakes.py`
  (`MalformedLLM`, `WrongNumberLLM`, `FlakyLLM`, `SlowLLM`) but no tests using them yet.
- No DAG, no DAG tests, no live-Ollama test, no seed script (see `docs/LIMITATIONS.md`).

## Debugging drills worth practising
Break the stale gate order in `Ledger.accept` and find it with `-k lin01`; break `find_quote` and look at `-k sec06`;
widen the retrieval filter in `QdrantStore.search` and look at `-m security`.
