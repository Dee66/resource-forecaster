"""
CDK Stack for Resource Forecaster

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
    aws_lambda_python_alpha as lambda_python,
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
    aws_sns as sns
)
from constructs import Construct


class ForecasterStack(Stack):
    """Main CDK stack for Resource Forecaster infrastructure."""

    def __init__(
        self, 
        scope: Construct, 
        construct_id: str,
        environment: str = "dev",
        **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.environment = environment
        
        # Create VPC with private subnets only
        self.vpc = self._create_vpc()
        
        # Create S3 bucket for model artifacts and data
        self.model_bucket = self._create_model_bucket()
        
        # Create DynamoDB table for job tracking
        self.job_table = self._create_job_table()
        
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
        
        # Create outputs
        self._create_outputs()

    def _create_vpc(self) -> ec2.Vpc:
        """Create VPC with private subnets and VPC endpoints."""
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
        
        cdk.Tags.of(vpc).add("App", "ResourceForecaster")
        cdk.Tags.of(vpc).add("Environment", self.environment)
        cdk.Tags.of(vpc).add("CostCenter", "MLOps")
        
        return vpc

    def _create_model_bucket(self) -> s3.Bucket:
        """Create S3 bucket for model artifacts and data storage."""
        bucket = s3.Bucket(
            self, "ModelBucket",
            bucket_name=f"resource-forecaster-models-{self.environment}-{self.account}",
            versioning=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.RETAIN if self.environment == "prod" else RemovalPolicy.DESTROY,
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
        
        cdk.Tags.of(bucket).add("App", "ResourceForecaster")
        cdk.Tags.of(bucket).add("Environment", self.environment)
        cdk.Tags.of(bucket).add("CostCenter", "MLOps")
        
        return bucket

    def _create_job_table(self) -> dynamodb.Table:
        """Create DynamoDB table for batch job tracking."""
        table = dynamodb.Table(
            self, "JobTable",
            table_name=f"forecaster-jobs-{self.environment}",
            partition_key=dynamodb.Attribute(
                name="job_id",
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="created_at",
                type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN if self.environment == "prod" else RemovalPolicy.DESTROY,
            point_in_time_recovery=True if self.environment == "prod" else False,
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
        
        cdk.Tags.of(table).add("App", "ResourceForecaster")
        cdk.Tags.of(table).add("Environment", self.environment)
        cdk.Tags.of(table).add("CostCenter", "MLOps")
        
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
        
        # Same permissions as Lambda role for task role
        for statement in self.lambda_role.policy_document.statements:
            task_role.add_to_policy(statement)
            
        return task_role

    def _create_prediction_lambda(self) -> lambda_.Function:
        """Create Lambda function for real-time predictions."""
        function = lambda_python.PythonFunction(
            self, "PredictionFunction",
            entry="lambda/prediction",
            runtime=lambda_.Runtime.PYTHON_3_11,
            index="handler.py",
            handler="lambda_handler",
            timeout=Duration.minutes(5),
            memory_size=1024,
            role=self.lambda_role,
            vpc=self.vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            environment={
                "MODEL_BUCKET": self.model_bucket.bucket_name,
                "JOB_TABLE": self.job_table.table_name,
                "ENVIRONMENT": self.environment,
                "LOG_LEVEL": "INFO"
            },
            log_retention=logs.RetentionDays.ONE_MONTH,
            reserved_concurrent_executions=50 if self.environment == "prod" else 10
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
        function = lambda_python.PythonFunction(
            self, "BatchFunction",
            entry="lambda/batch",
            runtime=lambda_.Runtime.PYTHON_3_11,
            index="handler.py",
            handler="lambda_handler",
            timeout=Duration.minutes(15),
            memory_size=2048,
            role=self.lambda_role,
            vpc=self.vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            environment={
                "MODEL_BUCKET": self.model_bucket.bucket_name,
                "JOB_TABLE": self.job_table.table_name,
                "ENVIRONMENT": self.environment,
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
            cluster_name=f"forecaster-{self.environment}",
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
                    "ENVIRONMENT": self.environment
                }
            ),
            memory_limit_mib=4096,
            cpu=2048,
            desired_count=1 if self.environment == "dev" else 2,
            public_load_balancer=False,
            service_name=f"forecaster-batch-{self.environment}"
        )
        
        # Configure health check
        service.target_group.configure_health_check(
            path="/health",
            healthy_http_codes="200"
        )
        
        # Configure auto scaling
        scaling = service.service.auto_scale_task_count(
            min_capacity=1,
            max_capacity=10 if self.environment == "prod" else 3
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
            rest_api_name=f"Resource Forecaster API - {self.environment}",
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
            api_key_name=f"forecaster-key-{self.environment}"
        )
        
        # Create usage plan
        usage_plan = api.add_usage_plan(
            "ForecasterUsagePlan",
            name=f"forecaster-usage-{self.environment}",
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
        
        high_anomaly_parallel.branch(
            send_alert_task
        ).branch(
            update_budgets_task
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
                send_alert_task.next(
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
                    log_group_name=f"/aws/stepfunctions/forecaster-{self.environment}",
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

    def _create_monitoring(self):
        """Create CloudWatch dashboards and alarms."""
        # Create dashboard
        dashboard = cloudwatch.Dashboard(
            self, "ForecasterDashboard",
            dashboard_name=f"ResourceForecaster-{self.environment}"
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
            topic_name=f"forecaster-alerts-{self.environment}",
            display_name=f"Resource Forecaster Alerts ({self.environment})"
        )
        
        # Daily forecast generation - 6 AM UTC (low-cost window)
        daily_rule = events.Rule(
            self, "DailyForecast",
            rule_name=f"forecaster-daily-{self.environment}",
            schedule=events.Schedule.cron(hour="6", minute="0"),  # 6 AM daily
            description="Daily cost forecast and optimization analysis"
        )
        
        daily_rule.add_target(
            targets.SfnStateMachine(
                self.step_function,
                input=events.RuleTargetInput.from_object({
                    "job_type": "forecast",
                    "job_name": f"daily-forecast-{self.environment}",
                    "parameters": {
                        "forecast_days": 30,
                        "include_recommendations": True
                    },
                    "environment": self.environment
                })
            )
        )
        
        # Weekly budget review - Monday 8 AM UTC
        weekly_budget_rule = events.Rule(
            self, "WeeklyBudgetReview",
            rule_name=f"forecaster-weekly-budget-{self.environment}",
            schedule=events.Schedule.cron(hour="8", minute="0", week_day="MON"),
            description="Weekly budget review and optimization analysis"
        )
        
        weekly_budget_rule.add_target(
            targets.SfnStateMachine(
                self.step_function,
                input=events.RuleTargetInput.from_object({
                    "job_type": "budget_review",
                    "job_name": f"weekly-budget-review-{self.environment}",
                    "parameters": {
                        "review_period_days": 7
                    },
                    "environment": self.environment
                })
            )
        )
        
        # Monthly optimization report - 1st of month 9 AM UTC
        monthly_report_rule = events.Rule(
            self, "MonthlyOptimization",
            rule_name=f"forecaster-monthly-{self.environment}",
            schedule=events.Schedule.cron(hour="9", minute="0", day="1"),
            description="Monthly cost optimization report"
        )
        
        monthly_report_rule.add_target(
            targets.SfnStateMachine(
                self.step_function,
                input=events.RuleTargetInput.from_object({
                    "job_type": "optimization_report",
                    "job_name": f"monthly-optimization-{self.environment}",
                    "parameters": {
                        "report_period_days": 30
                    },
                    "environment": self.environment
                })
            )
        )
        
        # Weekly model retraining - Sunday 2 AM UTC
        weekly_retrain_rule = events.Rule(
            self, "WeeklyRetrain",
            rule_name=f"forecaster-retrain-{self.environment}",
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
        if self.environment == "prod":
            anomaly_check_rule = events.Rule(
                self, "AnomalyCheck",
                rule_name=f"forecaster-anomaly-{self.environment}",
                schedule=events.Schedule.rate(Duration.minutes(30)),
                description="Real-time cost anomaly detection"
            )
            
            anomaly_check_rule.add_target(
                targets.SfnStateMachine(
                    self.step_function,
                    input=events.RuleTargetInput.from_object({
                        "job_type": "anomaly_check",
                        "job_name": f"anomaly-check-{self.environment}",
                        "parameters": {
                            "lookback_hours": 2
                        },
                        "environment": self.environment
                    })
                )
            )
        
        # Store alerts topic for access by Lambda functions
        self.alerts_topic = alerts_topic
        
        # Grant Step Functions permission to invoke Lambda functions
        self.prediction_lambda.grant_invoke(self.step_function.role)
        self.batch_lambda.grant_invoke(self.step_function.role)

    def _create_outputs(self):
        """Create CloudFormation outputs."""
        CfnOutput(
            self, "APIEndpoint",
            value=self.api.url,
            description="API Gateway endpoint URL"
        )
        
        CfnOutput(
            self, "ModelBucket",
            value=self.model_bucket.bucket_name,
            description="S3 bucket for model artifacts"
        )
        
        CfnOutput(
            self, "JobTable",
            value=self.job_table.table_name,
            description="DynamoDB table for job tracking"
        )
        
        CfnOutput(
            self, "VPCId",
            value=self.vpc.vpc_id,
            description="VPC ID for the forecaster infrastructure"
        )
        
        CfnOutput(
            self, "StateMachineArn",
            value=self.step_function.state_machine_arn,
            description="Step Functions state machine ARN"
        )