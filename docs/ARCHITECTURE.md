# Architecture (short)

```
client --X-API-Key--> FastAPI (app/main.py)
                        |  middleware: request-id, body-size cap (413)      error envelope: app/core/errors.py
                        v
   deps.get_principal: sha256(key) -> KeyRecord -> Principal(tenant, groups, scopes)   <- the ONLY source of the tenant
                        |
        +---------------+-------------------------+
        v                                         v
   POST /ingest  (docs:write)                POST /ask (ask:read)
   IngestionService                          AskService
     Ledger.accept   (SQLite, source of truth)   Retriever.retrieve
       stale -> duplicate -> conflict -> new        1. embed question
     chunk + embed                                  2. Qdrant ANN query WITH tenant+ACL filter
     upsert vectors (pending)                       3. per-hit ledger re-check (tenant, active version, ACL)
     Ledger.activate (supersede old)             build_messages  (random boundary, untrusted framing, aliases S1..Sn,
     delete older vectors                                          invisible characters stripped)
                                                 LLM (Ollama, JSON-Schema constrained output)
   GET /documents/{id}[/lineage]                 verify_answer  (aliases, verbatim quotes, numbers, conflict)
     tenant + ACL, uniform 404                   sanitize       (images, links, URLs, invisible characters)
```

## Decisions to defend
1. **The ledger is the source of truth.** SQLite records `latest_accepted_version` and `active_version` per tenant and document.
   The vector store is a derived index: a leftover point of an old version is never served because every hit is re-checked.
2. **Stale check before the idempotency check** (`Ledger.accept`). An Airflow retry replays the identical payload, so "same
   key already seen?" must not be asked before "is this older than what we accepted?".
3. **Filters are enforced in code, not in the prompt.** `TenantScope` is a capability only `Principal.scope()` can mint and
   every store method needs one; the ANN filter, the ledger re-check and the ACL rule (`can_read`) give the same decision
   (`SEC-22`). The prompt is treated as attacker-influenced, so the model is only ever shown authorised text.
4. **The model is not trusted.** The server extracts excerpts from the stored chunk and only accepts quotes that appear
   verbatim in a cited, authorised source; invented aliases, quotes and numbers lead to abstention.
5. **Ingestion is crash-safe by ordering:** write vectors as `pending`, activate in the ledger, then delete older vectors.

## Modules
| Path | Role |
|---|---|
| `app/api`, `app/core` | routes, auth deps, config, error envelope, middleware |
| `app/modules/ingestion` | ledger, chunking, ingestion service |
| `app/modules/retrieval` | embedder (Ollama or hash for tests), Qdrant store, retriever |
| `app/modules/tenancy` | `Principal`, `TenantScope`, `can_read` |
| `app/modules/generation` | prompt, Ollama client, verifier, sanitiser, normaliser |
| `dags/` | Airflow DAG (static-tested, not run under a scheduler) |
| `finetune/` | dataset, leakage check, QLoRA train, generate, eval, report |

## Trade-offs
Qdrant runs in embedded local mode and SQLite is single-node (one writer); both are enough for the assessment and
swappable behind the `VectorStore` protocol and the `Ledger` class. See `docs/LIMITATIONS.md` for what is not covered.
