"""
CDK Stack for resource-forecaster

Deploys cost forecasting infrastructure with VPC-only deployment,
least-privilege IAM, and comprehensive monitoring.
"""

from typing import Dict, Any, List
import aws_cdk as cdk
from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    CfnOutput
)
from aws_cdk import (
    aws_lambda as lambda_,
    aws_apigateway as apigw,
    aws_s3 as s3,
    aws_dynamodb as dynamodb,
    aws_iam as iam,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_logs as logs,
    aws_cloudwatch as cloudwatch,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as sfn_tasks,
    aws_events as events,
    aws_events_targets as targets,
    aws_secretsmanager as secrets,
    aws_ssm as ssm,
    aws_sns as sns,
    aws_sns_subscriptions as subscriptions,
    aws_cloudwatch_actions as cloudwatch_actions
)
try:
    # Try importing the alpha Python lambda construct; if not available,
    # set lambda_python to None and fall back to the stable implementation.
    from aws_cdk import aws_lambda_python_alpha as lambda_python  # type: ignore
except Exception:
    lambda_python = None
from constructs import Construct
from aws_cdk import aws_sagemaker as sagemaker


class ForecasterStack(Stack):
    """Main CDK stack for Resource Forecaster infrastructure."""

    def __init__(
        self, 
        scope: Construct, 
        construct_id: str,
        environment: str = "dev",
        model_bucket_name: str | None = None,
        data_bucket_name: str | None = None,
        vpc_id: str | None = None,
        **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Use a different attribute name to avoid collision with CDK's built-in
        # `environment` property on Stack.
        self.env_name = environment
        
        # Create VPC with private subnets only
        self.vpc = self._create_vpc(existing_vpc_id=vpc_id)

        # Create S3 bucket for model artifacts and data
        self.model_bucket = self._create_model_bucket(override_name=model_bucket_name)
        # Create S3 bucket for historical data with lifecycle policies
        self.historical_bucket = self._create_historical_bucket(override_name=data_bucket_name)
        
        # Create DynamoDB table for job tracking
        self.job_table = self._create_job_table()
        # Create DynamoDB table for rightsizing audit events
        self.audit_table = self._create_audit_table()
        
        # Create IAM roles
        self.lambda_role = self._create_lambda_role()
        self.ecs_role = self._create_ecs_role()
        
        # Create Lambda functions
        self.prediction_lambda = self._create_prediction_lambda()
        self.batch_lambda = self._create_batch_processing_lambda()
        
        # Create ECS cluster for long-running tasks
        self.ecs_cluster = self._create_ecs_cluster()
        self.batch_service = self._create_batch_service()
        
        # Create API Gateway
        self.api = self._create_api_gateway()
        
        # Create Step Functions for orchestration
        self.step_function = self._create_step_function()
        
        # Create CloudWatch monitoring
        self._create_monitoring()
        
        # Create scheduled jobs
        self._create_scheduled_jobs()

        # Optional SageMaker endpoint (synth-only unless enabled)
        enable_sm = cdk.CfnParameter(self, "EnableSageMaker",
                                     type="String",
                                     default="false",
                                     allowed_values=["true", "false"],
                                     description="Enable SageMaker endpoint creation (set true to create).")

        sm_condition = cdk.CfnCondition(self, "EnableSageMakerCondition",
                                        expression=cdk.Fn.condition_equals(enable_sm.value_as_string, "true"))

        image_param = cdk.CfnParameter(self, "SageMakerImageUri", type="String", default="", description="ECR image URI for SageMaker inference container")
        model_s3_param = cdk.CfnParameter(self, "SageMakerModelS3Uri", type="String", default="", description="S3 URI for model artifact (s3://...)")
        role_param = cdk.CfnParameter(self, "SageMakerExecutionRoleArn", type="String", default="", description="Execution role ARN for SageMaker")
        endpoint_name_param = cdk.CfnParameter(self, "SageMakerEndpointName", type="String", default=f"forecaster-sm-{self.env_name}", description="SageMaker endpoint name")

        primary_container = {"Image": image_param.value_as_string}
        if model_s3_param.value_as_string:
            primary_container["ModelDataUrl"] = model_s3_param.value_as_string

        sm_model = sagemaker.CfnModel(self, "SageMakerModel",
                                       execution_role_arn=role_param.value_as_string,
                                       primary_container=primary_container)
        sm_model.cfn_options.condition = sm_condition

        endpoint_config = sagemaker.CfnEndpointConfig(
            self,
            "SageMakerEndpointConfig",
            endpoint_config_name=f"{endpoint_name_param.value_as_string}-config",
            production_variants=[
                sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                    variant_name="AllTraffic",
                    model_name=sm_model.ref,
                    serverless_config=sagemaker.CfnEndpointConfig.ServerlessConfigProperty(
                        memory_size_in_mb=4096,
                        max_concurrency=4,
                    ),
                )
            ],
        )
        endpoint_config.cfn_options.condition = sm_condition

        sm_endpoint = sagemaker.CfnEndpoint(self, "SageMakerEndpoint",
                                            endpoint_name=endpoint_name_param.value_as_string,
                                            endpoint_config_name=endpoint_config.ref)
        sm_endpoint.cfn_options.condition = sm_condition
        
        # Create outputs
        self._create_outputs()

    def _create_vpc(self, existing_vpc_id: str | None = None) -> ec2.Vpc:
        """Create VPC with private subnets and VPC endpoints.

        If `existing_vpc_id` is provided, import the VPC via lookup instead of
        creating a new VPC. This allows infra to be parameterized for envs
        that supply an existing VPC.
        """
        if existing_vpc_id:
            # Import existing VPC by id (no network calls at synth; CDK may resolve during synth)
            return ec2.Vpc.from_lookup(self, "ImportedVPC", vpc_id=existing_vpc_id)  # type: ignore[arg-type]

        vpc = ec2.Vpc(
            self, "ForecasterVPC",
            max_azs=2,
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    subnet_type=ec2.SubnetType.PUBLIC,
                    name="Public",
                    cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    name="Private",
                    cidr_mask=24
                )
            ]
        )
        
        # VPC Endpoints for AWS services
        vpc.add_gateway_endpoint(
            "S3Endpoint",
            service=ec2.GatewayVpcEndpointAwsService.S3
        )
        
        vpc.add_interface_endpoint(
            "DynamoDBEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.DYNAMODB
        )
        
        vpc.add_interface_endpoint(
            "SecretsManagerEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.SECRETS_MANAGER
        )
        
        vpc.add_interface_endpoint(
            "CloudWatchEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS
        )
        
        cdk.Tags.of(vpc).add("App", "resource-forecaster")
        cdk.Tags.of(vpc).add("Environment", self.env_name)
        cdk.Tags.of(vpc).add("CostCenter", "MLOps")
        
        return vpc

    def _create_model_bucket(self, override_name: str | None = None) -> s3.Bucket:
        """Create S3 bucket for model artifacts and data storage.

        If `override_name` is provided, it will be used as the bucket name.
        """
        bucket_name = override_name or f"resource-forecaster-models-{self.env_name}-{self.account}"

        bucket = s3.Bucket(
            self, "ModelBucket",
            bucket_name=bucket_name,
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.RETAIN if self.env_name == "prod" else RemovalPolicy.DESTROY,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="DeleteOldVersions",
                    enabled=True,
                    noncurrent_version_expiration=Duration.days(90)
                ),
                s3.LifecycleRule(
                    id="ArchiveOldData",
                    enabled=True,
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INFREQUENT_ACCESS,
                            transition_after=Duration.days(30)
                        ),
                        s3.Transition(
                            storage_class=s3.StorageClass.GLACIER,
                            transition_after=Duration.days(90)
                        )
                    ]
                )
            ]
        )
        
        cdk.Tags.of(bucket).add("App", "resource-forecaster")
        cdk.Tags.of(bucket).add("Environment", self.env_name)
        cdk.Tags.of(bucket).add("CostCenter", "MLOps")
        
        return bucket

    def _create_historical_bucket(self, override_name: str | None = None) -> s3.Bucket:
        """Create S3 bucket for historical data with retention/lifecycle rules.

        Rules:
        - Transition to INFREQUENT_ACCESS after 30 days
        - Transition to GLACIER after 365 days
        - Expire (delete) objects after 1825 days (5 years)
        """
        bucket_name = override_name or f"resource-forecaster-historical-{self.env_name}-{self.account}"

        bucket = s3.Bucket(
            self, "HistoricalDataBucket",
            bucket_name=bucket_name,
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.RETAIN if self.env_name == "prod" else RemovalPolicy.DESTROY,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="HistoricalDataLifecycle",
                    enabled=True,
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INFREQUENT_ACCESS,
                            transition_after=Duration.days(30)
                        ),
                        s3.Transition(
                            storage_class=s3.StorageClass.GLACIER,
                            transition_after=Duration.days(365)
                        )
                    ],
                    expiration=Duration.days(1825)
                )
            ]
            )
        
        cdk.Tags.of(bucket).add("App", "resource-forecaster")
        cdk.Tags.of(bucket).add("Environment", self.env_name)
        cdk.Tags.of(bucket).add("CostCenter", "MLOps")

        return bucket

    def _create_job_table(self) -> dynamodb.Table:
        """Create DynamoDB table for batch job tracking."""
        table = dynamodb.Table(
            self, "JobTable",
            table_name=f"forecaster-jobs-{self.env_name}",
            partition_key=dynamodb.Attribute(
                name="job_id",
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="created_at",
                type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN if self.env_name == "prod" else RemovalPolicy.DESTROY,
            point_in_time_recovery=True if self.env_name == "prod" else False,
            time_to_live_attribute="ttl"
        )
        
        # Add GSI for status queries
        table.add_global_secondary_index(
            index_name="StatusIndex",
            partition_key=dynamodb.Attribute(
                name="status",
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="created_at",
                type=dynamodb.AttributeType.STRING
            )
        )
        
        cdk.Tags.of(table).add("App", "resource-forecaster")
        cdk.Tags.of(table).add("Environment", self.env_name)
        cdk.Tags.of(table).add("CostCenter", "MLOps")
        
        return table


    def _create_audit_table(self) -> dynamodb.Table:
        """Create DynamoDB table for rightsizing audit events."""
        table = dynamodb.Table(
            self, "RightsizingAuditTable",
            table_name=f"rightsizing-audit-{self.env_name}",
            partition_key=dynamodb.Attribute(
                name="recommendation_id",
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp",
                type=dynamodb.AttributeType.NUMBER
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN if self.env_name == "prod" else RemovalPolicy.DESTROY,
            point_in_time_recovery=True if self.env_name == "prod" else False,
            time_to_live_attribute="expires_at"
        )

        cdk.Tags.of(table).add("App", "resource-forecaster")
        cdk.Tags.of(table).add("Environment", self.env_name)
        cdk.Tags.of(table).add("CostCenter", "MLOps")

        # Output the table name
        CfnOutput(self, "AuditTableOutput", value=table.table_name, description="DynamoDB table for rightsizing audit events")

        return table

    def _create_lambda_role(self) -> iam.Role:
        """Create IAM role for Lambda functions with least-privilege access."""
        role = iam.Role(
            self, "LambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaVPCAccessExecutionRole")
            ]
        )
        
        # S3 permissions for model artifacts
        role.add_to_policy(iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            resources=[
                self.model_bucket.bucket_arn,
                f"{self.model_bucket.bucket_arn}/*"
            ]
        ))
        
        # DynamoDB permissions
        role.add_to_policy(iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "dynamodb:GetItem",
                "dynamodb:PutItem",
                "dynamodb:UpdateItem",
                "dynamodb:DeleteItem",
                "dynamodb:Query",
                "dynamodb:Scan"
            ],
            resources=[
                self.job_table.table_arn,
                f"{self.job_table.table_arn}/index/*"
            ]
        ))
        
        # Cost Explorer permissions
        role.add_to_policy(iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "ce:GetCostAndUsage",
                "ce:GetDimensionValues",
                "ce:GetReservationCoverage",
                "ce:GetReservationPurchaseRecommendation",
                "ce:GetReservationUtilization",
                "ce:GetSavingsPlansUtilization",
                "ce:GetSavingsPlansUtilizationDetails",
                "ce:GetSavingsPlansPurchaseRecommendation",
                "ce:GetRightsizingRecommendation"
            ],
            resources=["*"]
        ))
        
        # CloudWatch permissions
        role.add_to_policy(iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "cloudwatch:GetMetricData",
                "cloudwatch:GetMetricStatistics",
                "cloudwatch:ListMetrics"
            ],
            resources=["*"]
        ))
        
        # Secrets Manager permissions
        role.add_to_policy(iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "secretsmanager:GetSecretValue"
            ],
            resources=[f"arn:aws:secretsmanager:{self.region}:{self.account}:secret:forecaster/*"]
        ))
        
        return role

    def _create_ecs_role(self) -> iam.Role:
        """Create IAM role for ECS tasks."""
        task_role = iam.Role(
            self, "ECSTaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com")
        )
        
        execution_role = iam.Role(
            self, "ECSExecutionRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AmazonECSTaskExecutionRolePolicy")
            ]
        )
        
        # Grant similar permissions to the Lambda role for task role (explicitly)
        # S3 permissions for model artifacts
        task_role.add_to_policy(iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            resources=[
                self.model_bucket.bucket_arn,
                f"{self.model_bucket.bucket_arn}/*"
            ]
        ))

        # DynamoDB permissions
        task_role.add_to_policy(iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "dynamodb:GetItem",
                "dynamodb:PutItem",
                "dynamodb:UpdateItem",
                "dynamodb:DeleteItem",
                "dynamodb:Query",
                "dynamodb:Scan"
            ],
            resources=[
                self.job_table.table_arn,
                f"{self.job_table.table_arn}/index/*"
            ]
        ))

        # Cost Explorer permissions
        task_role.add_to_policy(iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "ce:GetCostAndUsage",
                "ce:GetDimensionValues",
                "ce:GetReservationCoverage",
                "ce:GetReservationPurchaseRecommendation",
                "ce:GetReservationUtilization",
                "ce:GetSavingsPlansUtilization",
                "ce:GetSavingsPlansUtilizationDetails",
                "ce:GetSavingsPlansPurchaseRecommendation",
                "ce:GetRightsizingRecommendation"
            ],
            resources=["*"]
        ))

        # CloudWatch permissions
        task_role.add_to_policy(iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "cloudwatch:GetMetricData",
                "cloudwatch:GetMetricStatistics",
                "cloudwatch:ListMetrics"
            ],
            resources=["*"]
        ))

        # Secrets Manager permissions
        task_role.add_to_policy(iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "secretsmanager:GetSecretValue"
            ],
            resources=[f"arn:aws:secretsmanager:{self.region}:{self.account}:secret:forecaster/*"]
        ))

        return task_role

    def _create_prediction_lambda(self) -> lambda_.Function:
        """Create Lambda function for real-time predictions."""
        function = self._create_python_lambda(
            id="PredictionFunction",
            entry="lambda/prediction",
            index="handler.py",
            handler="lambda_handler",
            timeout=Duration.minutes(5),
            memory_size=1024,
            reserved_concurrent_executions=50 if self.env_name == "prod" else 10,
            environment={
                "MODEL_BUCKET": self.model_bucket.bucket_name,
                "JOB_TABLE": self.job_table.table_name,
                "ENVIRONMENT": self.env_name,
                "LOG_LEVEL": "INFO"
            }
        )
        
        # Create CloudWatch alarms
        function.metric_errors().create_alarm(
            self, "PredictionErrors",
            threshold=10,
            evaluation_periods=2,
            alarm_description="High error rate in prediction function"
        )
        
        function.metric_duration().create_alarm(
            self, "PredictionDuration",
            threshold=Duration.seconds(30).to_seconds(),
            evaluation_periods=2,
            alarm_description="High latency in prediction function"
        )
        
        return function

    def _create_batch_processing_lambda(self) -> lambda_.Function:
        """Create Lambda function for batch job orchestration."""
        function = self._create_python_lambda(
            id="BatchFunction",
            entry="lambda/batch",
            index="handler.py",
            handler="lambda_handler",
            timeout=Duration.minutes(15),
            memory_size=2048,
            environment={
                "MODEL_BUCKET": self.model_bucket.bucket_name,
                "JOB_TABLE": self.job_table.table_name,
                "ENVIRONMENT": self.env_name,
                "LOG_LEVEL": "INFO"
            },
            log_retention=logs.RetentionDays.ONE_MONTH
        )
        
        return function

    def _create_ecs_cluster(self) -> ecs.Cluster:
        """Create ECS cluster for long-running batch tasks."""
        cluster = ecs.Cluster(
            self, "ForecasterCluster",
            vpc=self.vpc,
            cluster_name=f"forecaster-{self.env_name}",
            container_insights=True
        )
        
        return cluster

    def _create_batch_service(self) -> ecs_patterns.ApplicationLoadBalancedFargateService:
        """Create ECS Fargate service for batch processing."""
        service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self, "BatchService",
            cluster=self.ecs_cluster,
            task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                image=ecs.ContainerImage.from_asset("docker/batch"),
                container_port=8000,
                task_role=self.ecs_role,
                environment={
                    "MODEL_BUCKET": self.model_bucket.bucket_name,
                    "JOB_TABLE": self.job_table.table_name,
                    "ENVIRONMENT": self.env_name
                }
            ),
            memory_limit_mib=4096,
            cpu=2048,
            desired_count=1 if self.env_name == "dev" else 2,
            public_load_balancer=False,
            service_name=f"forecaster-batch-{self.env_name}"
        )
        
        # Configure health check
        service.target_group.configure_health_check(
            path="/health",
            healthy_http_codes="200"
        )
        
        # Configure auto scaling
        scaling = service.service.auto_scale_task_count(
            min_capacity=1,
            max_capacity=10 if self.env_name == "prod" else 3
        )
        
        scaling.scale_on_cpu_utilization(
            "CpuScaling",
            target_utilization_percent=70
        )
        
        scaling.scale_on_memory_utilization(
            "MemoryScaling",
            target_utilization_percent=80
        )
        
        return service

    def _create_api_gateway(self) -> apigw.RestApi:
        """Create API Gateway for REST endpoints."""
        api = apigw.RestApi(
            self, "ForecasterAPI",
            rest_api_name=f"resource-forecaster-api-{self.env_name}",
            description="Cost forecasting and optimization recommendations API",
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=apigw.Cors.ALL_ORIGINS,
                allow_methods=apigw.Cors.ALL_METHODS,
                allow_headers=["Content-Type", "Authorization"]
            ),
            endpoint_types=[apigw.EndpointType.REGIONAL]
        )
        
        # Add API key for authentication
        api_key = api.add_api_key(
            "ForecasterAPIKey",
            api_key_name=f"forecaster-key-{self.env_name}"
        )
        
        # Create usage plan
        usage_plan = api.add_usage_plan(
            "ForecasterUsagePlan",
            name=f"forecaster-usage-{self.env_name}",
            throttle=apigw.ThrottleSettings(
                rate_limit=1000,
                burst_limit=2000
            ),
            quota=apigw.QuotaSettings(
                limit=100000,
                period=apigw.Period.MONTH
            )
        )
        
        usage_plan.add_api_key(api_key)
        usage_plan.add_api_stage(
            stage=api.deployment_stage
        )
        
        # Prediction endpoints
        predict_resource = api.root.add_resource("predict")
        predict_resource.add_method(
            "POST",
            apigw.LambdaIntegration(self.prediction_lambda),
            api_key_required=True
        )
        
        # Batch endpoints
        batch_resource = api.root.add_resource("batch")
        batch_resource.add_method(
            "POST",
            apigw.LambdaIntegration(self.batch_lambda),
            api_key_required=True
        )
        
        # Job status endpoints
        jobs_resource = api.root.add_resource("jobs")
        job_resource = jobs_resource.add_resource("{job_id}")
        job_resource.add_method(
            "GET",
            apigw.LambdaIntegration(self.batch_lambda),
            api_key_required=True
        )
        
        return api

    def _create_step_function(self) -> sfn.StateMachine:
        """Create Step Functions for workflow orchestration."""
        # Define tasks
        collect_data_task = sfn_tasks.LambdaInvoke(
            self, "CollectDataTask",
            lambda_function=self.prediction_lambda,
            payload=sfn.TaskInput.from_object({
                "action": "collect_data",
                "input.$": "$"
            }),
            result_path="$.collect_result"
        )
        
        train_model_task = sfn_tasks.LambdaInvoke(
            self, "TrainModelTask",
            lambda_function=self.batch_lambda,
            payload=sfn.TaskInput.from_object({
                "action": "train_model",
                "input.$": "$"
            }),
            result_path="$.train_result"
        )
        
        generate_forecast_task = sfn_tasks.LambdaInvoke(
            self, "GenerateForecastTask",
            lambda_function=self.prediction_lambda,
            payload=sfn.TaskInput.from_object({
                "action": "generate_forecast",
                "input.$": "$"
            }),
            result_path="$.forecast_result"
        )
        
        evaluate_forecast_task = sfn_tasks.LambdaInvoke(
            self, "EvaluateForecastTask",
            lambda_function=self.prediction_lambda,
            payload=sfn.TaskInput.from_object({
                "action": "evaluate_forecast",
                "input.$": "$"
            }),
            result_path="$.evaluation_result"
        )
        
        send_alert_task = sfn_tasks.LambdaInvoke(
            self, "SendAlertTask",
            lambda_function=self.prediction_lambda,
            payload=sfn.TaskInput.from_object({
                "action": "send_alerts",
                "input.$": "$"
            }),
            result_path="$.alert_result"
        )
        
        update_budgets_task = sfn_tasks.LambdaInvoke(
            self, "UpdateBudgetsTask",
            lambda_function=self.prediction_lambda,
            payload=sfn.TaskInput.from_object({
                "action": "update_budgets",
                "input.$": "$"
            }),
            result_path="$.budget_result"
        )
        
        # Define workflow with parallel execution where appropriate
        data_collection = collect_data_task
        
        model_training_parallel = sfn.Parallel(
            self, "ModelTrainingParallel",
            comment="Train models and generate forecasts in parallel"
        )
        
        model_training_parallel.branch(
            train_model_task
        ).branch(
            generate_forecast_task
        )
        
        # Evaluation and action workflow
        evaluation_choice = sfn.Choice(self, "EvaluateAnomalies")
        
        high_anomaly_parallel = sfn.Parallel(
            self, "HighAnomalyActions",
            comment="Send alerts and update budgets for high anomalies"
        )
        # Create distinct task instances for branches to avoid state reuse
        send_alert_task_high = sfn_tasks.LambdaInvoke(
            self, "SendAlertTaskHigh",
            lambda_function=self.prediction_lambda,
            payload=sfn.TaskInput.from_object({
                "action": "send_alerts",
                "input.$": "$"
            }),
            result_path="$.alert_result"
        )

        update_budgets_task_high = sfn_tasks.LambdaInvoke(
            self, "UpdateBudgetsTaskHigh",
            lambda_function=self.prediction_lambda,
            payload=sfn.TaskInput.from_object({
                "action": "update_budgets",
                "input.$": "$"
            }),
            result_path="$.budget_result"
        )

        high_anomaly_parallel.branch(
            send_alert_task_high
        ).branch(
            update_budgets_task_high
        )
        
        # Define the complete workflow
        definition = data_collection.next(
            model_training_parallel
        ).next(
            evaluate_forecast_task
        ).next(
            evaluation_choice.when(
                sfn.Condition.number_greater_than("$.evaluation_result.Payload.data.alert_data.anomaly_score", 0.8),
                high_anomaly_parallel.next(
                    sfn.Succeed(self, "HighAnomalySuccess", comment="Successfully handled high anomaly")
                )
            ).when(
                sfn.Condition.number_greater_than("$.evaluation_result.Payload.data.alert_data.anomaly_score", 0.5),
                # Use a dedicated medium-priority alert task to avoid reuse
                sfn_tasks.LambdaInvoke(
                    self, "SendAlertTaskMedium",
                    lambda_function=self.prediction_lambda,
                    payload=sfn.TaskInput.from_object({
                        "action": "send_alerts",
                        "input.$": "$"
                    }),
                    result_path="$.alert_result"
                ).next(
                    sfn.Succeed(self, "MediumAnomalySuccess", comment="Successfully handled medium anomaly")
                )
            ).otherwise(
                update_budgets_task.next(
                    sfn.Succeed(self, "NormalFlowSuccess", comment="Successfully completed normal workflow")
                )
            )
        )
        
        # Create state machine
        state_machine = sfn.StateMachine(
            self, "ForecasterWorkflow",
            definition=definition,
            timeout=Duration.hours(4),
            logs=sfn.LogOptions(
                destination=logs.LogGroup(
                    self, "WorkflowLogGroup",
                    log_group_name=f"/aws/stepfunctions/forecaster-{self.env_name}",
                    retention=logs.RetentionDays.ONE_MONTH
                ),
                level=sfn.LogLevel.ALL
            )
        )
        
        # Create CloudWatch alarms for state machine
        state_machine.metric_failed().create_alarm(
            self, "WorkflowFailures",
            threshold=1,
            evaluation_periods=1,
            alarm_description="Step Functions workflow failures"
        )
        
        state_machine.metric_timed_out().create_alarm(
            self, "WorkflowTimeouts",
            threshold=1,
            evaluation_periods=1,
            alarm_description="Step Functions workflow timeouts"
        )
        
        return state_machine

    def _create_python_lambda(self, *, id: str, entry: str, index: str = "handler.py", handler: str = "lambda_handler", timeout: Duration = Duration.minutes(5), memory_size: int = 128, environment: dict | None = None, reserved_concurrent_executions: int | None = None, log_retention: logs.RetentionDays | None = None, use_shared_role: bool = True) -> lambda_.Function:
        """Create a Python Lambda function using the alpha construct if available,
        otherwise fall back to the stable aws_lambda.Function with a local asset.
        """
        # Prefer the aws_lambda_python_alpha.PythonFunction (if available) for
        # convenient dependency packaging. Otherwise, create a generic
        # aws_lambda.Function using the contents of `entry` as an asset.
        if lambda_python is not None:
            # If use_shared_role is False, allow the construct to create its own role
            kwargs = dict(
                entry=entry,
                runtime=lambda_.Runtime.PYTHON_3_11,
                index=index,
                handler=handler,
                timeout=timeout,
                memory_size=memory_size,
                vpc=self.vpc,
                vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
                environment=environment or {},
                log_retention=log_retention or logs.RetentionDays.ONE_MONTH,
                reserved_concurrent_executions=reserved_concurrent_executions
            )
            if use_shared_role:
                kwargs["role"] = self.lambda_role
            return lambda_python.PythonFunction(self, id, **kwargs)

        # Fallback: create a generic Lambda with code from an asset
        code = lambda_.Code.from_asset(entry)
        # Build kwargs for the fallback Function
        fn_kwargs = dict(
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler=f"{index.rstrip('.py')}.{handler}",
            code=code,
            timeout=timeout,
            memory_size=memory_size,
            vpc=self.vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            environment=environment or {}
        )
        if use_shared_role:
            fn_kwargs["role"] = self.lambda_role

        fn = lambda_.Function(self, id, **fn_kwargs)

        if reserved_concurrent_executions is not None:
            # Use the low-level CfnFunction to set ReservedConcurrentExecutions
            cfn_fn = fn.node.default_child
            if hasattr(cfn_fn, "add_override"):
                cfn_fn.add_override("Properties.ReservedConcurrentExecutions", reserved_concurrent_executions)

        return fn

    def _create_monitoring(self):
        """Create CloudWatch dashboards and alarms."""
        # Create dashboard
        dashboard = cloudwatch.Dashboard(
            self, "ForecasterDashboard",
            dashboard_name=f"resource-forecaster-{self.env_name}"
        )
        
        # Lambda metrics
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Lambda Function Invocations",
                left=[
                    self.prediction_lambda.metric_invocations(),
                    self.batch_lambda.metric_invocations()
                ]
            ),
            cloudwatch.GraphWidget(
                title="Lambda Function Errors",
                left=[
                    self.prediction_lambda.metric_errors(),
                    self.batch_lambda.metric_errors()
                ]
            ),
            cloudwatch.GraphWidget(
                title="Lambda Function Duration",
                left=[
                    self.prediction_lambda.metric_duration(),
                    self.batch_lambda.metric_duration()
                ]
            )
        )
        
        # API Gateway metrics
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="API Gateway Requests",
                left=[
                    self.api.metric_count(),
                    self.api.metric_client_error(),
                    self.api.metric_server_error()
                ]
            ),
            cloudwatch.GraphWidget(
                title="API Gateway Latency",
                left=[
                    self.api.metric_latency(),
                    self.api.metric_integration_latency()
                ]
            )
        )
        
        # DynamoDB metrics
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="DynamoDB Operations",
                left=[
                    self.job_table.metric("ConsumedReadCapacityUnits"),
                    self.job_table.metric("ConsumedWriteCapacityUnits")
                ]
            )
        )

    def _create_scheduled_jobs(self):
        """Create scheduled jobs for automated forecasting."""
        # Create SNS topic for alerts
        alerts_topic = sns.Topic(
            self, "AlertsTopic",
            topic_name=f"resource-forecaster-alerts-{self.env_name}",
            display_name=f"resource-forecaster-alerts-{self.env_name}"
        )

        # Create a safe shutdown Lambda to handle non-prod shutdown recommendations
        # The Lambda runs in DRY_RUN mode by default to avoid accidental resource termination.
        shutdown_lambda = self._create_python_lambda(
            id="ShutdownLambda",
            entry="lambda/shutdown",
            index="handler.py",
            handler="lambda_handler",
            timeout=Duration.minutes(5),
            memory_size=128,
            environment={
                "DRY_RUN": "true",
                "ENVIRONMENT": self.env_name,
                # Audit table name will be provided so lambda can persist recommendation events
                "AUDIT_TABLE_NAME": self.audit_table.table_name
            },
            log_retention=logs.RetentionDays.ONE_WEEK
        )

        # Subscribe the shutdown Lambda to the alerts topic so alarms can trigger it.
        alerts_topic.add_subscription(subscriptions.LambdaSubscription(shutdown_lambda))
        # Grant the shutdown lambda permission to write to the audit table
        self.audit_table.grant_write_data(shutdown_lambda)
        # Also grant prediction and batch lambdas permission to write audit events
        # This allows these functions to persist recommendation/audit events directly
        self.audit_table.grant_write_data(self.prediction_lambda)
        self.audit_table.grant_write_data(self.batch_lambda)
        
        # Daily forecast generation - 6 AM UTC (low-cost window)
        daily_rule = events.Rule(
            self, "DailyForecast",
                rule_name=f"forecaster-daily-{self.env_name}",
            schedule=events.Schedule.cron(hour="6", minute="0"),  # 6 AM daily
            description="Daily cost forecast and optimization analysis"
        )
        
        daily_rule.add_target(
            targets.SfnStateMachine(
                self.step_function,
                input=events.RuleTargetInput.from_object({
                    "job_type": "forecast",
                    "job_name": f"daily-forecast-{self.env_name}",
                    "parameters": {
                        "forecast_days": 30,
                        "include_recommendations": True
                    },
                    "environment": self.env_name
                })
            )
        )
        
        # Weekly budget review - Monday 8 AM UTC
        weekly_budget_rule = events.Rule(
            self, "WeeklyBudgetReview",
                rule_name=f"forecaster-weekly-budget-{self.env_name}",
            schedule=events.Schedule.cron(hour="8", minute="0", week_day="MON"),
            description="Weekly budget review and optimization analysis"
        )
        
        weekly_budget_rule.add_target(
            targets.SfnStateMachine(
                self.step_function,
                input=events.RuleTargetInput.from_object({
                    "job_type": "budget_review",
                    "job_name": f"weekly-budget-review-{self.env_name}",
                    "parameters": {
                        "review_period_days": 7
                    },
                    "environment": self.env_name
                })
            )
        )
        
        # Monthly optimization report - 1st of month 9 AM UTC
        monthly_report_rule = events.Rule(
            self, "MonthlyOptimization",
                rule_name=f"forecaster-monthly-{self.env_name}",
            schedule=events.Schedule.cron(hour="9", minute="0", day="1"),
            description="Monthly cost optimization report"
        )
        
        monthly_report_rule.add_target(
            targets.SfnStateMachine(
                self.step_function,
                input=events.RuleTargetInput.from_object({
                    "job_type": "optimization_report",
                    "job_name": f"monthly-optimization-{self.env_name}",
                    "parameters": {
                        "report_period_days": 30
                    },
                    "environment": self.env_name
                })
            )
        )
        
        # Weekly model retraining - Sunday 2 AM UTC
        weekly_retrain_rule = events.Rule(
            self, "WeeklyRetrain",
            rule_name=f"forecaster-retrain-{self.env_name}",
            schedule=events.Schedule.cron(hour="2", minute="0", week_day="SUN"),  # 2 AM Sundays
            description="Weekly model retraining"
        )
        
        weekly_retrain_rule.add_target(
            targets.LambdaFunction(
                self.batch_lambda,
                event=events.RuleTargetInput.from_object({
                    "action": "retrain_models",
                    "lookback_days": 90
                })
            )
        )
        
        # Real-time anomaly detection - every 30 minutes (prod only)
        if self.env_name == "prod":
            anomaly_check_rule = events.Rule(
                self, "AnomalyCheck",
                rule_name=f"forecaster-anomaly-{self.env_name}",
                schedule=events.Schedule.rate(Duration.minutes(30)),
                description="Real-time cost anomaly detection"
            )
            
            anomaly_check_rule.add_target(
                targets.SfnStateMachine(
                    self.step_function,
                    input=events.RuleTargetInput.from_object({
                        "job_type": "anomaly_check",
                        "job_name": f"anomaly-check-{self.env_name}",
                        "parameters": {
                            "lookback_hours": 2
                        },
                        "environment": self.env_name
                    })
                )
            )
        
        # Store alerts topic for access by Lambda functions
        self.alerts_topic = alerts_topic

        # Create RMSE drift alarm on custom metric and attach SNS action
        try:
            rmse_metric = cloudwatch.Metric(
                namespace="resource-forecaster/Metrics",
                metric_name="RMSE",
                dimensions_map={"Environment": self.environment},
                statistic="Average",
                period=Duration.minutes(5)
            )

            rmse_alarm = cloudwatch.Alarm(
                self, "RmseDriftAlarm",
                metric=rmse_metric,
                threshold=0.05,
                evaluation_periods=3,
                alarm_description=f"RMSE drift alarm for {self.env_name} (threshold=0.05)",
                alarm_name=f"resource-forecaster-RMSE-Drift-{self.env_name}"
            )

            # When alarm triggers, notify the alerts SNS topic (and thus the Lambda/subscribers)
            rmse_alarm.add_alarm_action(cloudwatch_actions.SnsAction(alerts_topic))
            # Also create a low-priority OK action to silence in certain conditions if desired
        except Exception:
            # If CloudWatch metric not yet present during synth, continue gracefully
            pass

        # Create a lightweight retrain Lambda that will be triggered by the RMSE SNS topic
        # It will start the retrain workflow (Step Function) when invoked.
        retrain_lambda = self._create_python_lambda(
            id="RmseRetrainLambda",
            entry="lambda/retrain",
            index="handler.py",
            handler="lambda_handler",
            timeout=Duration.minutes(1),
            memory_size=128,
            environment={
                "STATE_MACHINE_ARN": self.step_function.state_machine_arn,
                "ENVIRONMENT": self.env_name
            },
            log_retention=logs.RetentionDays.ONE_WEEK,
            use_shared_role=False
        )

        # Grant retrain lambda permissions to start the state machine
        self.step_function.grant_start_execution(retrain_lambda)

        # Subscribe retrain lambda to alerts topic so RMSE alarm will trigger retrain
        alerts_topic.add_subscription(subscriptions.LambdaSubscription(retrain_lambda))
        
        # Grant Step Functions permission to invoke Lambda functions
        self.prediction_lambda.grant_invoke(self.step_function.role)
        self.batch_lambda.grant_invoke(self.step_function.role)

        # --- Nightly reconciliation Lambda to verify recommendations were actioned
        reconciler_lambda = self._create_python_lambda(
            id="ReconcilerLambda",
            entry="lambda/reconciler",
            index="handler.py",
            handler="lambda_handler",
            timeout=Duration.minutes(2),
            memory_size=128,
            environment={
                "AUDIT_TABLE_NAME": self.audit_table.table_name,
                "ALERTS_TOPIC_ARN": alerts_topic.topic_arn,
                "ENVIRONMENT": self.env_name
            },
            log_retention=logs.RetentionDays.ONE_WEEK
        )

        # Grant read access to the audit table and publish permission to alerts topic
        self.audit_table.grant_read_data(reconciler_lambda)
        alerts_topic.grant_publish(reconciler_lambda)
        # Allow Reconciler Lambda to put CloudWatch metrics (by attaching a policy)
        reconciler_lambda.add_to_role_policy(iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=["cloudwatch:PutMetricData"],
            resources=["*"]
        ))

        # Schedule nightly (2 AM UTC) reconciliation
        reconcile_rule = events.Rule(
            self, "NightlyReconcile",
            rule_name=f"forecaster-nightly-reconcile-{self.env_name}",
            schedule=events.Schedule.cron(hour="2", minute="0"),
            description="Nightly reconciliation of audit recommendations vs actual resources"
        )

        reconcile_rule.add_target(targets.LambdaFunction(reconciler_lambda))

        # Output reconciler Lambda name
        CfnOutput(self, "ReconcilerLambdaName", value=reconciler_lambda.function_name, description="Reconciliation Lambda name")

        # Create CloudWatch alarm for reconciliation missing count > 0
        missing_metric = cloudwatch.Metric(
            namespace="resource-forecaster/Monitoring",
            metric_name="ReconciliationMissingCount",
            dimensions_map={"Environment": self.env_name},
            statistic="Sum",
            period=Duration.minutes(60)
        )

        reconcile_alarm = cloudwatch.Alarm(
            self, "ReconcileMissingAlarm",
            metric=missing_metric,
            threshold=0,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            alarm_description="Reconciliation detected missing recommended implementations",
            alarm_name=f"resource-forecaster-Reconciliation-Missing-{self.env_name}"
        )

        # Notify via SNS
        reconcile_alarm.add_alarm_action(cloudwatch_actions.SnsAction(alerts_topic))

    def _create_outputs(self):
        """Create CloudFormation outputs."""
        CfnOutput(
            self, "APIEndpointOutput",
            value=self.api.url,
            description="API Gateway endpoint URL"
        )

        CfnOutput(
            self, "ModelBucketOutput",
            value=self.model_bucket.bucket_name,
            description="S3 bucket for model artifacts"
        )

        CfnOutput(
            self, "JobTableOutput",
            value=self.job_table.table_name,
            description="DynamoDB table for job tracking"
        )

        CfnOutput(
            self, "VPCIdOutput",
            value=self.vpc.vpc_id,
            description="VPC ID for the forecaster infrastructure"
        )

        CfnOutput(
            self, "StateMachineArnOutput",
            value=self.step_function.state_machine_arn,
            description="Step Functions state machine ARN"
        )