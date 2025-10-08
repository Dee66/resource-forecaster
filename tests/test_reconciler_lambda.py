import json
import time
import importlib.machinery
import importlib.util
import pathlib

# Load the reconciler lambda handler module directly from file
reconciler_path = str(pathlib.Path(__file__).parent.parent / 'lambda' / 'reconciler' / 'handler.py')
loader = importlib.machinery.SourceFileLoader('reconciler_handler', reconciler_path)
spec = importlib.util.spec_from_loader(loader.name, loader)
reconciler = importlib.util.module_from_spec(spec)
loader.exec_module(reconciler)


class DummyDDB:
    def __init__(self, items):
        self._items = items

    def scan(self, TableName=None, Limit=None):
        return {"Items": self._items}


class DummySNS:
    def __init__(self):
        self.published = []

    def publish(self, TopicArn=None, Subject=None, Message=None):
        self.published.append({"TopicArn": TopicArn, "Subject": Subject, "Message": Message})
        return {"MessageId": "m-1"}


class DummyEC2:
    def describe_instances(self, InstanceIds=None):
        # pretend instance i-exists exists, others missing
        if InstanceIds and InstanceIds[0] == 'i-exists':
            return {"Reservations": [{"Instances": [{"InstanceId": 'i-exists'}]}]}
        return {"Reservations": []}


class DummyCW:
    def __init__(self):
        self.metrics = []

    def put_metric_data(self, Namespace=None, MetricData=None):
        self.metrics.append({"Namespace": Namespace, "MetricData": MetricData})
        return {}


def test_reconciler_no_items():
    clients = {"dynamodb": DummyDDB([]), "sns": DummySNS(), "ec2": DummyEC2()}
    # set env
    import os
    os.environ["AUDIT_TABLE_NAME"] = "rightsizing-audit-dev"
    os.environ["ALERTS_TOPIC_ARN"] = "arn:aws:sns:us-east-1:123:alerts"
    res = reconciler.lambda_handler({}, None, clients=clients)
    assert res["status"] == "ok"
    assert res["total"] == 0


def test_reconciler_missing_and_publish(monkeypatch):
    items = [{"resource_id": {"S": "i-missing"}}, {"resource_id": {"S": "i-exists"}}]
    clients = {"dynamodb": DummyDDB(items), "sns": DummySNS(), "ec2": DummyEC2(), "cloudwatch": DummyCW(), "rds": None, "autoscaling": None}
    # set fake env via monkeypatch
    monkeypatch.setenv("AUDIT_TABLE_NAME", "rightsizing-audit-dev")
    monkeypatch.setenv("ALERTS_TOPIC_ARN", "arn:aws:sns:us-east-1:123:alerts")

    res = reconciler.lambda_handler({}, None, clients=clients)
    assert res["status"] == "ok"
    assert res["total"] == 2
    assert res["missing_count"] == 1
    # verify SNS published
    assert len(clients["sns"].published) == 1
    pub = clients["sns"].published[0]
    msg = json.loads(pub["Message"])
    assert msg["missing_count"] == 1
    # verify cloudwatch metric was written
    assert len(clients["cloudwatch"].metrics) == 1
    md = clients["cloudwatch"].metrics[0]["MetricData"]
    assert md[0]["Value"] == 1
