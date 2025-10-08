# FAQ

Q: How accurate are the forecasts?
A: The system uses RMSE and MAPE as primary metrics. A model fidelity guard in CI prevents promotion if RMSE > 5% or MAPE > 10%.

Q: How are recommendations applied safely?
A: Non-production shutdowns are performed in DRY_RUN by default. Audit events are persisted to DynamoDB. Critical recommendations require manual approval per the runbook.

Q: How is drift handled?
A: RMSE is emitted to CloudWatch and an alarm triggers an automated retrain starter that invokes the Step Function retrain pipeline.

Q: Where are artifacts stored?
A: Model artifacts and data are stored in S3 buckets. Historical data lifecycle is enforced (IA @30d, Glacier @365d, expire @1825d).

Q: Can I use this for other resources beyond EC2?
A: Yes — the recommendation engine is designed to be extended. Cost Explorer APIs support other services; the BudgetAlertManager can be extended similarly.
