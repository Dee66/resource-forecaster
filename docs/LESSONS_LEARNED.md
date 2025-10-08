# Lessons Learned & Future Enhancements

- Dont tightly couple Lambda roles with Step Function permissions when the Step Function references those Lambdas — it creates a CFN dependency cycle. Use a dedicated starter role for retrain triggers.
- CDK logical ids are hashed; tests should avoid exact logical id matching. Use substring/regex or look up by resource properties.
- Emit key ML metrics (RMSE, MAPE, R²) as CloudWatch metrics early — they become essential for drift detection and CI gating.

Future enhancements
- Add a reconciliation Lambda that runs nightly and publishes a compliance metric (recommendation implementation rate).
- Add a small UI showing recommendation status and audit trail.
- Explore XGBoost-based ensemble for non-linear relationships in the forecast.
