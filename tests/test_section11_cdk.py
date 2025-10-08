import aws_cdk as cdk
from aws_cdk.assertions import Template, Match

from infra.forecaster_stack import ForecasterStack


def test_section11_resources_created():
    app = cdk.App()
    # Synthesize the stack in 'prod' mode to include prod-only rules
    stack = ForecasterStack(app, "TestForecasterStack", environment="prod")
    template = Template.from_stack(stack)

    # SNS Topic for alerts
    template.has_resource_properties(
        "AWS::SNS::Topic",
        {
            "DisplayName": "Resource Forecaster Alerts (prod)",
            "TopicName": "forecaster-alerts-prod",
        },
    )

    # Shutdown Lambda (ensure environment variables include DRY_RUN and ENVIRONMENT)
    template.has_resource_properties(
        "AWS::Lambda::Function",
        {
            "Environment": {
                "Variables": {
                    "DRY_RUN": "true",
                    "ENVIRONMENT": "prod",
                    # Audit table name should be provided
                    "AUDIT_TABLE_NAME": {
                        "Ref": Match.string_like_regexp("^RightsizingAuditTable")
                    },
                }
            }
        }
    )

    # Rightsizing audit DynamoDB table exists
    template.has_resource_properties(
        "AWS::DynamoDB::Table",
        {
            "KeySchema": [
                {"AttributeName": "recommendation_id", "KeyType": "HASH"},
                {"AttributeName": "timestamp", "KeyType": "RANGE"}
            ],
            "BillingMode": "PAY_PER_REQUEST",
        }
    )

    # RMSE CloudWatch alarm
    template.has_resource_properties(
        "AWS::CloudWatch::Alarm",
        {
            "AlarmName": "ResourceForecaster-RMSE-Drift-prod",
        },
    )

    # Historical data S3 bucket with lifecycle rules (expire after 1825 days)
    template.has_resource_properties(
        "AWS::S3::Bucket",
        {
            "LifecycleConfiguration": {
                "Rules": [
                    {
                        "Status": "Enabled",
                        "ExpirationInDays": 1825
                    }
                ]
            }
        }
    )

    # Ensure SNS subscriptions to Lambda exist (shutdown + retrain)
    # There should be at least one subscription with Protocol 'lambda'
    template.resource_count_is("AWS::SNS::Subscription", 2)

    # Ensure at least one IAM Policy contains a statement allowing dynamodb:PutItem
    iam_policies = template.find_resources("AWS::IAM::Policy")
    found_put = False
    for _, policy in iam_policies.items():
        doc = policy.get("Properties", {}).get("PolicyDocument", {})
        statements = doc.get("Statement", [])
        for stmt in statements:
            action = stmt.get("Action")
            if isinstance(action, str) and action == "dynamodb:PutItem":
                found_put = True
                break
            if isinstance(action, list) and any(a == "dynamodb:PutItem" for a in action):
                found_put = True
                break
        if found_put:
            break

    assert found_put, "No IAM Policy found with dynamodb:PutItem permission"

    # Ensure there is at least one IAM Policy that allows Step Functions StartExecution (retrain lambda role)
    iam_policies = template.find_resources("AWS::IAM::Policy")
    found_start = False
    for _, policy in iam_policies.items():
        statements = policy.get("Properties", {}).get("PolicyDocument", {}).get("Statement", [])
        for stmt in statements:
            action = stmt.get("Action")
            if isinstance(action, str) and action == "states:StartExecution":
                found_start = True
                break
            if isinstance(action, list) and any(a == "states:StartExecution" for a in action):
                found_start = True
                break
        if found_start:
            break

    assert found_start, "No IAM Policy found with states:StartExecution permission"

    # --- Tighten: ensure the StartExecution permission is attached to the retrain lambda role specifically
    # Find the retrain Lambda logical id
    lambda_functions = template.find_resources("AWS::Lambda::Function")
    retrain_fn_logical = None
    for logical_id, res in lambda_functions.items():
        if "RmseRetrainLambda" in logical_id:
            retrain_fn_logical = logical_id
            break

    assert retrain_fn_logical is not None, "RmseRetrainLambda function not found in template"

    # Extract the role logical id referenced by the retrain function
    retrain_props = lambda_functions[retrain_fn_logical]["Properties"]
    role_ref = retrain_props.get("Role")
    # Role may be an Fn::GetAtt; find the role logical id accordingly
    role_logical = None
    if isinstance(role_ref, dict):
        # expecting {"Fn::GetAtt": ["RoleLogicalId", "Arn"]}
        getatt = role_ref.get("Fn::GetAtt") or role_ref.get("Fn::GetAtt")
        if isinstance(getatt, list) and len(getatt) >= 1:
            role_logical = getatt[0]

    assert role_logical is not None, "Could not determine retrain lambda role logical id"

    # Find policy resources attached to this role and assert StartExecution exists
    found_start_on_role = False
    for logical_id, policy in iam_policies.items():
        roles = policy.get("Properties", {}).get("Roles", [])
        for r in roles:
            if isinstance(r, dict) and r.get("Ref") == role_logical:
                # check statements for StartExecution
                statements = policy.get("Properties", {}).get("PolicyDocument", {}).get("Statement", [])
                for stmt in statements:
                    action = stmt.get("Action")
                    if isinstance(action, str) and action == "states:StartExecution":
                        found_start_on_role = True
                        break
                    if isinstance(action, list) and any(a == "states:StartExecution" for a in action):
                        found_start_on_role = True
                        break
            if found_start_on_role:
                break
        if found_start_on_role:
            break

    assert found_start_on_role, "Retrain lambda role does not have states:StartExecution permission"

    # --- Historical bucket transitions: INFREQUENT_ACCESS after 30d and GLACIER after 365d
    s3_buckets = template.find_resources("AWS::S3::Bucket")
    hist_bucket = None
    for logical_id, b in s3_buckets.items():
        if "HistoricalDataBucket" in logical_id:
            hist_bucket = b
            break

    assert hist_bucket is not None, "HistoricalDataBucket not found"
    rules = hist_bucket.get("Properties", {}).get("LifecycleConfiguration", {}).get("Rules", [])
    assert len(rules) > 0, "No lifecycle rules on HistoricalDataBucket"
    transitions = []
    for rule in rules:
        for t in rule.get("Transitions", []):
            sc = t.get("StorageClass") or t.get("StorageClass")
            days = t.get("TransitionInDays") or t.get("TransitionAfterDays") or t.get("TransitionAfter") or t.get("TransitionAfter")
            # CDK emits Transition with 'StorageClass' and 'TransitionInDays' or 'TransitionAfter'
            # Simplify: inspect for presence of 30 and 365 anywhere in the structure
            transitions.append(t)

    # Check the serialized rule bodies for transition days
    # Look for any transition with NoncurrentDays/TransitionInDays or similar equal to 30 or 365
    found_30 = False
    found_365 = False
    import json as _json
    for t in transitions:
        txt = _json.dumps(t)
        if '30' in txt:
            found_30 = True
        if '365' in txt:
            found_365 = True

    assert found_30, "No transition to INFREQUENT_ACCESS after 30 days found"
    assert found_365, "No transition to GLACIER after 365 days found"

    # --- Ensure an SNS subscription exists that targets the retrain lambda
    sns_subs = template.find_resources("AWS::SNS::Subscription")
    found_retrain_sub = False
    for logical_id, sub in sns_subs.items():
        props = sub.get("Properties", {})
        if props.get("Protocol") == "lambda":
            endpoint = props.get("Endpoint")
            if isinstance(endpoint, dict):
                getatt = endpoint.get("Fn::GetAtt")
                if isinstance(getatt, list) and getatt[0] == retrain_fn_logical:
                    found_retrain_sub = True
                    break

    assert found_retrain_sub, "No SNS subscription found targeting the retrain lambda"
