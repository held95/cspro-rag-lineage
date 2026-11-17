"""Who is calling. The tenant lives here and ONLY here.

``TenantScope`` is a capability object: every vector-store method requires one, and it can only be
created from an authenticated ``Principal`` - so code cannot query the shared collection without
saying (and being limited to) which tenant it acts for.
"""
from __future__ import annotations

from dataclasses import dataclass, field

_ISSUER = object()


@dataclass(frozen=True)
class TenantScope:
    tenant_id: str
    groups: frozenset[str]
    _issuer: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self._issuer is not _ISSUER:
            raise TypeError("TenantScope can only be created through Principal.scope()")


@dataclass(frozen=True)
class Principal:
    tenant_id: str
    subject: str
    groups: frozenset[str] = frozenset()
    scopes: frozenset[str] = frozenset()

    def scope(self) -> TenantScope:
        return TenantScope(self.tenant_id, self.groups, _ISSUER)
