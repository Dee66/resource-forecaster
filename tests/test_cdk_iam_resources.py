import aws_cdk as cdk
from aws_cdk.assertions import Template

from infra.forecaster_stack import ForecasterStack


def test_iam_policies_contain_expected_resources():
    app = cdk.App()
    stack = ForecasterStack(app, "TestForecasterStack", environment="prod")
    template = Template.from_stack(stack)

    # Find the Step Function and DynamoDB table logical ids to build expected ARNs
    sfn = template.find_resources("AWS::StepFunctions::StateMachine")
    assert len(sfn) == 1
    sfn_logical = list(sfn.keys())[0]

    ddb = template.find_resources("AWS::DynamoDB::Table")
    # Find the RightsizingAuditTable logical id
    audit_logical = None
    for k in ddb.keys():
        if 'RightsizingAuditTable' in k:
            audit_logical = k
            break

    assert audit_logical is not None, "RightsizingAuditTable not found"

    iam_policies = template.find_resources("AWS::IAM::Policy")

    found_sfn_resource = False
    found_ddb_resource = False

    for _, policy in iam_policies.items():
        statements = policy.get('Properties', {}).get('PolicyDocument', {}).get('Statement', [])
        for stmt in statements:
            res = stmt.get('Resource')
            # Resource may be list or string or intrinsic
            candidates = []
            if isinstance(res, list):
                candidates.extend(res)
            else:
                candidates.append(res)

            for r in candidates:
                # Case: Fn::GetAtt referencing state machine or Ref referencing table
                if isinstance(r, dict):
                    if r.get('Fn::GetAtt') and r['Fn::GetAtt'][0] == sfn_logical:
                        found_sfn_resource = True
                    if r.get('Ref') and r['Ref'] == audit_logical:
                        found_ddb_resource = True
                    # accept Fn::GetAtt entries that reference the audit table logical id substring
                    if r.get('Fn::GetAtt') and 'RightsizingAudit' in r['Fn::GetAtt'][0]:
                        found_ddb_resource = True
                    # also accept Ref pointing to the state machine logical id
                    if r.get('Ref') and r['Ref'] == sfn_logical:
                        found_sfn_resource = True
                    # Fn::Sub may contain literal stateMachine in the string
                    if r.get('Fn::Sub'):
                        sub = r.get('Fn::Sub')
                        if isinstance(sub, str) and 'stateMachine' in sub:
                            found_sfn_resource = True
                        if isinstance(sub, str) and 'rightsizing' in sub:
                            found_ddb_resource = True
                elif isinstance(r, str):
                    if 'stateMachine' in r:
                        found_sfn_resource = True
                    if 'rightsizing' in r or 'RightsizingAudit' in r:
                        found_ddb_resource = True

    assert found_sfn_resource, "No IAM policy references the Step Function resource"
    assert found_ddb_resource, "No IAM policy references the RightsizingAuditTable resource"
