"""LIN-01 / LIN-02: the assessment's central failure, reproduced and fixed.

The SAME test body runs against the naive pipeline (strict xfail = the bug is demonstrated and
must keep failing) and against the corrected pipeline (must pass).
Run:  pytest -m failure_repro -v
"""
import pytest

from tests.helpers import make_stack, manifest, principal
from tests.naive_pipeline import NaivePipeline

pytestmark = [pytest.mark.failure_repro]

V1 = "Refund Policy (v1). Customers may request a refund within 14 days of delivery."
V2 = "Refund Policy (v2). Customers may request a refund within 30 days of delivery."


class FixedAdapter:
    """Drives the real service and reads back what retrieval would actually serve."""

    def __init__(self, tmp_path):
        self.stack = make_stack(tmp_path)

    def ingest(self, tenant, doc_id, version, content):
        self.stack.service.ingest(principal(tenant).scope(), manifest(doc_id, version, content))

    def visible_versions(self, tenant, doc_id):
        ctx = self.stack.retriever.retrieve(principal(tenant), "refund within days of delivery", top_k=10)
        return {h.version for h in ctx.hits if h.doc_id == doc_id}


NAIVE_BUG = pytest.mark.xfail(strict=True, reason="naive pipeline reproduces the lineage bug")


@pytest.fixture(params=[pytest.param("naive", marks=NAIVE_BUG), "fixed"])
def pipeline(request, tmp_path):
    return NaivePipeline() if request.param == "naive" else FixedAdapter(tmp_path)


def test_lin01_retry_does_not_resurrect_stale_version(pipeline):
    pipeline.ingest("alpha", "KB-001", 1, V1)   # DAG run 1
    pipeline.ingest("alpha", "KB-001", 2, V2)   # DAG run 2 supersedes v1
    pipeline.ingest("alpha", "KB-001", 1, V1)   # retry / clear of run 1 replays the OLD payload
    assert pipeline.visible_versions("alpha", "KB-001") == {2}


def test_lin02_superseding_in_one_tenant_never_touches_another(pipeline):
    pipeline.ingest("beta", "KB-001", 1, "Beta Retail refund policy: 7 days of delivery.")
    pipeline.ingest("alpha", "KB-001", 1, V1)
    pipeline.ingest("alpha", "KB-001", 2, V2)
    assert pipeline.visible_versions("beta", "KB-001") == {1}
