"""Document-level access rule, in Python. The vector store builds an equivalent filter for the
ANN query; tests assert both give identical decisions (SEC-22)."""
from __future__ import annotations

from app.modules.tenancy.principal import TenantScope


def can_read(scope: TenantScope, doc_tenant: str, visibility: str, acl_groups: list[str] | tuple[str, ...]) -> bool:
    if doc_tenant != scope.tenant_id:
        return False
    if visibility == "tenant":
        return True
    return bool(set(acl_groups) & set(scope.groups))
