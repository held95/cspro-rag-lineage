"""SEC-05b: the ledger re-check must also compare the chunk's tenant.

Found by a mutation drill: removing the `hit.tenant_id != principal.tenant_id` line of
`Retriever._authorized` was caught by NO test. The ledger lookup is keyed by the CALLER's tenant, so when a
broken store returns beta's chunk of `KB-001 v1` and alpha also has an active `KB-001 v1`, the ledger row
found is alpha's and the foreign chunk would be served. Only the explicit tenant comparison stops it."""
import pytest

from tests.helpers import make_stack, manifest, principal
from tests.security.controls_off import UnscopedStore

pytestmark = pytest.mark.security


def test_sec05b_foreign_chunk_with_same_doc_id_and_version_is_dropped(tmp_path):
    stack = make_stack(tmp_path)
    alpha, beta = principal("alpha"), principal("beta")
    stack.service.ingest(alpha.scope(), manifest("KB-001", 1, "Alpha refund window is 14 days."))
    stack.service.ingest(beta.scope(), manifest("KB-001", 1, "Beta returns are accepted within 7 days CANARY-BETA-7f3a9."))
    stack.retriever.store = UnscopedStore(stack.store)  # broken store: ignores the tenant scope

    query = "Beta returns accepted within 7 days CANARY-BETA-7f3a9"
    raw = stack.retriever.store.search(alpha.scope(), stack.embedder.embed([query], "query")[0], 10)
    assert "beta" in {h.tenant_id for h in raw}  # precondition: the broken store really returns the foreign chunk

    ctx = stack.retriever.retrieve(alpha, query, 10)
    assert {h.tenant_id for h in ctx.hits} <= {"alpha"}
    assert "CANARY-BETA-7f3a9" not in " ".join(h.text for h in ctx.hits)
    # the drop is audited
    events = stack.ledger.events("alpha", "KB-001")
    assert any(e["action"] == "security_drop" for e in events)
