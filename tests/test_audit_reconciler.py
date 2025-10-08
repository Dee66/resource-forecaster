from types import SimpleNamespace

from src.forecaster.audit_reconciler import reconcile_audit_with_resources


class DummyDB:
    def __init__(self, items):
        self._items = items

    def scan(self, TableName=None, Limit=None):
        return {'Items': self._items}


def test_reconcile_audit_with_resources():
    # Create items: one existing, one missing
    items = [
        {'resource_id': {'S': 'i-exists'}, 'recommendation_id': {'S': 'r1'}},
        {'resource_id': {'S': 'i-missing'}, 'recommendation_id': {'S': 'r2'}}
    ]

    db = DummyDB(items)

    def checker(rid):
        return rid == 'i-exists'

    summary = reconcile_audit_with_resources(db, 'rightsizing-audit-dev', checker)

    assert summary['total'] == 2
    assert summary['implemented'] == 1
    assert summary['missing_count'] == 1
    assert summary['missing'][0]['resource_id'] == 'i-missing'
