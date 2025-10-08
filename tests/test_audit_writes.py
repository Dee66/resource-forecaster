import time

from types import SimpleNamespace

from src.forecaster.workflow.budget_alert_manager import BudgetAlertManager, CostRecommendation


class DummyDynamo:
    def __init__(self):
        self.items = []

    def put_item(self, TableName=None, Item=None):
        self.items.append((TableName, Item))
        return {'ResponseMetadata': {'HTTPStatusCode': 200}}


def test_process_budget_alert_writes_audit_events(monkeypatch):
    # Create dummy config with audit_table_name
    cfg = SimpleNamespace(environment='dev', audit_table_name='rightsizing-audit-dev')

    manager = BudgetAlertManager(cfg)

    # Inject dummy dynamodb client
    dummy_db = DummyDynamo()
    manager.dynamodb = dummy_db

    # Prepare fake recommendations
    recs = [
        CostRecommendation('rightsizing', 'i-123', 200.0, 50.0, 0.9, 'Resize', 'high'),
        CostRecommendation('reserved_instance', 'm5.large', 500.0, 150.0, 0.95, 'Purchase RI', 'medium')
    ]

    # Monkeypatch get_cost_recommendations to return our recs
    monkeypatch.setattr(manager, 'get_cost_recommendations', lambda: recs)

    # Call process_budget_alert
    event = {'budgetName': 'test', 'threshold': 90, 'actualSpend': 1000, 'forecastedSpend': 1200}
    resp = manager.process_budget_alert(event)

    # Ensure audit writes occurred for both recommendations
    assert len(dummy_db.items) == 2
    table_names = {t for t, _ in dummy_db.items}
    assert 'rightsizing-audit-dev' in table_names
