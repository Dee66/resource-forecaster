import aws_cdk as cdk
from aws_cdk.assertions import Template

from infra.forecaster_stack import ForecasterStack


def test_iam_policies_attached_to_lambda_roles():
    app = cdk.App()
    stack = ForecasterStack(app, "TestForecasterStack", environment="prod")
    template = Template.from_stack(stack)

    # Map Lambda logical ids to expected function name fragments
    lambda_functions = template.find_resources("AWS::Lambda::Function")
    assert any("PredictionFunction" in k for k in lambda_functions.keys()), "PredictionFunction missing"
    assert any("BatchFunction" in k for k in lambda_functions.keys()), "BatchFunction missing"
    assert any("ShutdownLambda" in k for k in lambda_functions.keys()), "ShutdownLambda missing"
    assert any("RmseRetrainLambda" in k for k in lambda_functions.keys()), "RmseRetrainLambda missing"

    # Collect role logical ids referenced by each function
    fn_to_role = {}
    for logical_id, res in lambda_functions.items():
        props = res.get("Properties", {})
        role = props.get("Role")
        if isinstance(role, dict) and role.get("Fn::GetAtt"):
            fn_to_role[logical_id] = role["Fn::GetAtt"][0]

    iam_policies = template.find_resources("AWS::IAM::Policy")

    # For each of the important lambda roles, ensure there is at least one IAM::Policy that references it
    for fn_logical, role_logical in fn_to_role.items():
        found = False
        for _, policy in iam_policies.items():
            roles = policy.get("Properties", {}).get("Roles", [])
            for r in roles:
                if isinstance(r, dict) and r.get("Ref") == role_logical:
                    found = True
                    break
            if found:
                break
        assert found, f"No IAM::Policy found attached to role {role_logical} for function {fn_logical}"

    # Additionally, ensure that any policy that allows dynamodb:PutItem is attached to at least one of the lambda roles
    putitem_policies = []
    for _, policy in iam_policies.items():
        statements = policy.get("Properties", {}).get("PolicyDocument", {}).get("Statement", [])
        for stmt in statements:
            action = stmt.get("Action")
            if isinstance(action, str) and action == "dynamodb:PutItem":
                putitem_policies.append(policy)
            if isinstance(action, list) and any(a == "dynamodb:PutItem" for a in action):
                putitem_policies.append(policy)

    assert len(putitem_policies) > 0, "No policy with dynamodb:PutItem found"

    # Ensure at least one such policy is attached to a lambda role
    attached = False
    role_refs = set(fn_to_role.values())
    for policy in putitem_policies:
        roles = policy.get("Properties", {}).get("Roles", [])
        for r in roles:
            if isinstance(r, dict) and r.get("Ref") in role_refs:
                attached = True
                break
        if attached:
            break

    assert attached, "No dynamodb:PutItem policy found attached to any lambda role"
