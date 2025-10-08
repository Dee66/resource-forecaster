import aws_cdk as cdk
from aws_cdk.assertions import Template

from infra.forecaster_stack import ForecasterStack


def test_historical_bucket_tags_and_transitions():
    app = cdk.App()
    stack = ForecasterStack(app, "TestForecasterStack", environment="prod")
    template = Template.from_stack(stack)

    s3_buckets = template.find_resources("AWS::S3::Bucket")
    hist_bucket = None
    for logical_id, b in s3_buckets.items():
        if "HistoricalDataBucket" in logical_id:
            hist_bucket = b
            break

    assert hist_bucket is not None, "HistoricalDataBucket missing"

    props = hist_bucket.get("Properties", {})
    # Check tags exist on the bucket (App and Environment)
    tags = props.get("Tags", [])
    tag_keys = {t.get("Key") for t in tags}
    assert "App" in tag_keys and "Environment" in tag_keys, "Expected tags not present on historical bucket"

    # Check lifecycle transitions for explicit storage classes
    rules = props.get("LifecycleConfiguration", {}).get("Rules", [])
    assert len(rules) > 0, "No lifecycle rules present"

    storage_classes = set()
    for rule in rules:
        for t in rule.get("Transitions", []):
            storage_classes.add(t.get("StorageClass") or t.get("StorageClass"))

    # CDK may represent StorageClass as strings like 'GLACIER' or mapping; check substrings
    joined = " ".join([str(s).upper() for s in storage_classes if s])
    assert ("INFREQUENT" in joined or "STANDARD_IA" in joined or "INTELLIGENT_TIERING" in joined), "INFREQUENT_ACCESS transition missing"
    assert "GLACIER" in joined or "DEEP_ARCHIVE" in joined, "GLACIER transition missing"
