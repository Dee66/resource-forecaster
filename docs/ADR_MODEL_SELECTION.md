# ADR: Forecasting Model Selection & Error Budget

Status: Proposed

Context

- The Resource Forecaster project predicts costs and generates optimization recommendations (rightsizing, RIs, Savings Plans).
- Models considered: Prophet, ARIMA, simple linear models, ensembles.
- Key operational constraint: forecasts must be reliable enough to drive recommendations that affect cost/resources.

Decision

- Primary model: Prophet (Facebook / Meta Prophet) as the default base model for seasonal/time-series forecasting.
  - Rationale: Good out-of-the-box seasonality handling, holiday support, and familiar to stakeholders.
- Ensemble: ensemble predictions combine Prophet + simple linear + ARIMA in production-sensitive paths when ensemble performance improves RMSE consistently in backtests.
- Error budget / quality gate:
  - CI fails model package promotion if validation RMSE > 0.05 (5%) or MAPE > 10%.
  - For incremental retrain events, require validation RMSE to improve or remain within a rolling window of 5%.
- Monitoring & drift:
  - RMSE is emitted to CloudWatch and alarmed (RMSE drift alarm already present in CDK).
  - If alarm triggers, retrain starter Lambda initiates retrain pipeline and alerts owners.

Consequences

- Using Prophet minimizes feature engineering for seasonality but requires careful hyperparameter tuning for non-seasonal workloads.
- Ensemble adds compute but improves worst-case RMSE; enable in prod if backtests show lift.

Alternatives considered

- Pure ARIMA (good for non-seasonal/short horizon), but more brittle across noisy AWS billing data.
- Pure ML regressors (XGBoost): higher engineering cost but potentially better at capturing non-linear relationships. Consider as future enhancement.

Reviewers

- Team: FinOps, ML Eng, Platform


```
Decision recorded: 2025-10-07
```
