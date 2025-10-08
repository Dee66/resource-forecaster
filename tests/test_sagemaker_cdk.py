import aws_cdk as cdk
from aws_cdk.assertions import Template

from infra.forecaster_stack import ForecasterStack


def test_sagemaker_resources_guarded():
    app = cdk.App()
    stack = ForecasterStack(app, "TestForecasterStack", environment="dev")
    template = Template.from_stack(stack)

    # Ensure SageMaker model/config/endpoint resources are in the template
    models = template.find_resources("AWS::SageMaker::Model")
    configs = template.find_resources("AWS::SageMaker::EndpointConfig")
    endpoints = template.find_resources("AWS::SageMaker::Endpoint")

    assert len(models) >= 1, "No SageMaker Model resource found in template"
    assert len(configs) >= 1, "No SageMaker EndpointConfig resource found in template"
    assert len(endpoints) >= 1, "No SageMaker Endpoint resource found in template"

    # Each SageMaker resource should be guarded by the EnableSageMakerCondition
    for _, res in models.items():
        assert res.get("Condition") == "EnableSageMakerCondition"
    for _, res in configs.items():
        assert res.get("Condition") == "EnableSageMakerCondition"
    for _, res in endpoints.items():
        assert res.get("Condition") == "EnableSageMakerCondition"


def test_enable_sagemaker_parameter_default_false():
    app = cdk.App()
    stack = ForecasterStack(app, "TestForecasterStack", environment="dev")
    template = Template.from_stack(stack)

    tpl = template.to_json()
    params = tpl.get("Parameters", {})
    assert "EnableSageMaker" in params
    enable_param = params["EnableSageMaker"]
    assert enable_param.get("Default") == "false"
