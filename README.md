# Syskit Data Scientist Task - Submission README

This repository contains an end-to-end solution for the Syskit Data Scientist take-home task. The work covers data quality review, tenant-level feature engineering, customer health scoring, churn modeling, marketing channel analysis, a stakeholder-facing dashboard, and a short executive summary.

## Repository Structure
- `notebooks/01_data_quality_check.ipynb`
  - Profiles the source data, runs structural and temporal data quality checks, and creates a cleaned subscriptions table.
- `notebooks/02_health_score_and_feature_store.ipynb`
  - Analyzes pre-churn behavior and builds the tenant-level feature store, health score, and action flags.
- `notebooks/03_churn_model.ipynb`
  - Trains and evaluates a 90-day churn model and interprets the coefficients.
- `notebooks/04_acquisition_channel_analysis.ipynb`
  - Evaluates acquisition channels on retention and growth potential and exports a marketing chart.
- `src/syskit_utils.py`
  - Shared utility functions for loading the source data.
- `results/`
  - Exported analytical outputs used by the notebooks and dashboard.
- `dashboard/streamlit_app.py`
  - Streamlit dashboard source code.
- `docs/executive_summary.pdf`
  - Leadership-facing summary for the VP of Customer Success and VP of Sales.

## Run Order
1. `notebooks/01_data_quality_check.ipynb`
2. `notebooks/02_health_score_and_feature_store.ipynb`
3. `notebooks/03_churn_model.ipynb`
4. `notebooks/04_acquisition_channel_analysis.ipynb`


## Required Input
- `notebooks/saas_dataset.sqlite`

## Generated Outputs
- `data/subscriptions_clean.csv`
- `results/tenant_feature_store_snapshot_2024_06_30.csv`
- `results/tenant_scores_snapshot_2024_06_30.csv`
- `results/prechurn_event_drops.csv`
- `results/acquisition_channel_stay_grow.png`

## Tools Used
- `Python`
  - Used for data processing, feature engineering, modeling, and dashboard logic.
- `Jupyter Notebooks`
  - Used to organize the work into reproducible analysis stages.
- `SQLite`
  - Used as the source datastore through `saas_dataset.sqlite`.
- `pandas` and `numpy`
  - Used for cleaning, aggregation, feature engineering, and general analysis.
- `scikit-learn`
  - Used for logistic regression, preprocessing, and cross-validation.
- `matplotlib`
  - Used for the marketing and growth visualizations.
- `Altair`
  - Used for interactive charts in the Streamlit dashboard.
- `Streamlit`
  - Used to build the stakeholder-facing dashboard.
- `Codex`
  - Used for coding and writing assistance.

## 1. Data Pipeline and Model

The brief asked for a reproducible pipeline that integrates the provided sources, handles data quality issues explicitly, and produces a clean analytical layer. This is implemented primarily in `notebooks/01_data_quality_check.ipynb` and `notebooks/02_health_score_and_feature_store.ipynb`.

### Data Quality Review
- The source model is structurally strong. There are no duplicate business keys and no broken foreign-key relationships across the six core tables.
- Some date fields (`activity_date`, `churn_date`, `last_seen_at`) extend beyond the telemetry cutoff (2024-06-30). This is not inherently problematic, but it requires a careful approach to avoid data leakage during modeling.
- The main issues are temporal. Product telemetry appears both before `registered_at` and after `last_seen_at` and `renewal_date`. In addition, some tenants are marked as active (`churned==0`) even though they are already past `renewal_date`.

### Cleaning Decisions
- I did not modify the raw source tables.
- I treated telemetry as the most reliable behavioral signal and did not rely on raw `registered_at` and `last_seen_at` in downstream analysis.
- I derived `next_renewal_date` for problematic renewal records based on the observed 365-day distance between `contract_start_date` and `renewal_date`.
- I saved the cleaned subscriptions table to `data/subscriptions_clean.csv`.

### Analytical Grain
The central analytical layer is a tenant-level snapshot as of `2024-06-30`, which is the end of the telemetry window. This is the right grain because Customer Success and Sales operate at the account level.

The derived features use only information available on or before the telemetry cutoff date, so they can be used to build a churn model that predicts churn after the snapshot date.

The exported feature store is:
- `results/tenant_feature_store_snapshot_2024_06_30.csv`

## 2. Customer Analytics

The brief asked for a customer health score, an at-risk cohort, and a usage trend view over the last 8 to 12 weeks. These are implemented in `notebooks/02_health_score_and_feature_store.ipynb` and shown in the dashboard https://syskit.streamlit.app/.

### Approach
I analyzed tenants that churned before the telemetry cutoff to understand how behavior changes before churn and how far before contract renewal churn typically happens. Those findings were then converted into a health score and operational flags for tenants that are still active at the snapshot date.

I did not build a trial-to-paid funnel analysis because stage transitions are not clear in the provided data.

### Main Findings
- Churn tends to occur before contract expiration rather than exactly at renewal.
- Median churn timing is `45.5` days before renewal.
- The interquartile range is `16.8` to `75.8` days before renewal.
- Usage declines across all events before churn, but the steepest drops are concentrated in high-value product actions such as:
  - `sensitivity_label_applied`
  - `license_recommendation_applied`
  - `pp_sync_completed`
  - `risky_workspace_resolved`
  - `policy_created`

These findings are reflected in the health score design by giving more weight to value-realization signals than to lightweight activity.

### Health Score and Action Flags
- `health_score_0_100`
  - Weighted percentile score based on value-event momentum, usage breadth, and CRM interaction quality.
- `health_segment`
  - Segments tenants into `critical`, `watchlist`, `stable`, and `strong`.
- `at_risk_flag`
  - Flags weak-health tenants with near-term renewal pressure.
- `quiet_account_flag`
  - Flags low-activity tenants with weak momentum and no recent CS contact.
- `expansion_candidate_flag`
  - Flags healthier lower-tier tenants with increasing activity.

### Current Snapshot Counts
From `results/tenant_scores_snapshot_2024_06_30.csv`:
- Active tenants: `450`
- `at_risk_flag`: `48`
- `quiet_account_flag`: `11`
- `expansion_candidate_flag`: `50`
- Health segments:
  - `critical`: `38`
  - `watchlist`: `176`
  - `stable`: `190`
  - `strong`: `46`

### Usage Trend
The 12-week usage trend is shown in the dashboard and can be filtered by customer segment. It supports both total usage and value-event usage views.

## 3. Predictive Model

The brief required at least one predictive model, a clear explanation of metric choice, and a discussion of limitations. This is implemented in `notebooks/03_churn_model.ipynb`.

### Modeling Choices
- Prediction target: 90-day churn risk for tenants active on `2024-06-30`
- Model family: logistic regression
- Validation: 5-fold stratified cross-validation

I chose churn prediction because churn is the cleanest business outcome in the dataset with an observable timestamp. Other candidate targets such as upsell or trial conversion do not have equally reliable labels here.

I used logistic regression because the active-snapshot sample is small and the model is stable, interpretable, and appropriate for a first operational scoring layer. Cross-validation makes better use of the available data than a simple single train/test split.

Another advantage of logistic regression is its interpretability (see the Coefficient Interpretation section below).


### Evaluation Output
- Snapshot-active tenants: `450`
- Positive 90-day churners: `23`
- ROC-AUC: `0.926`
- Precision at top decile: `0.356`
- Recall at top decile: `0.696`

### Why These Metrics
Accuracy is not useful here because churners are a minority class. The most relevant metrics are:
- `ROC-AUC`
  - Measures how well the model ranks churners above non-churners across thresholds.
- `Precision at top decile`
  - Measures how many tenants in the highest-risk 10% actually churn. This directly reflects the quality of the outreach list that CS would prioritize first.
- `Recall at top decile`
  - Measures how many of all true churners are captured inside that highest-risk 10%. This reflects coverage: how much churn risk we catch with limited intervention bandwidth.

### Coefficient Interpretation
Features most associated with higher churn risk:
- `no_response_touch_rate_90d`
- `negative_touch_rate_90d`
- `users_total`
- `days_to_renewal_<180`
- `arr`

Features most associated with lower churn risk:
- `tenure_days`
- `positive_touch_rate_90d`
- `value_share_30d`
- `event_types_used_30d`
- `health_score_0_100`

This suggests that weak CRM engagement quality (no response / negative outcomes) and near-term renewal pressure are risk signals. Also larger companies (more users, larger arr) have higher churn risk.

Longer tenure, healthier product-value usage patterns, and positive CRM interactions are protective.

### Limitations
1. The dataset is small, with only 23 churners, which means model performance should be closely monitored as more data becomes available.
2. The model is trained and evaluated on tenants active at one snapshot date (`2024-06-30`), so it might not generalize well to future periods.
3. No external holdout test set. Due to small dataset I used out-of-fold predictions for validation, but there is no final untouched dataset for confirmation.
4. The model predicts whether a customer will churn within the next 90 days. This horizon may be too long, since earlier analysis showed that meaningful behavior changes typically begin about 45 days before churn.


## 4. Marketing and Growth Analysis

The brief asked which acquisition channels bring in customers who stay and grow. This is implemented in `notebooks/04_acquisition_channel_analysis.ipynb`.

### Approach
I used two complementary views:
- Historical churn comparison by acquisition channel
- Snapshot-forward comparison based on:
  - `retention_90d`
  - `grow_potential` from `expansion_signal`

The combined score is:
- `stay_and_grow_score = 0.6 * retention_90d + 0.4 * grow_potential`

### Main Findings
- Both the historical and snapshot-forward views support the conclusion that **referral** is the top channel for bringing in customers who stay, while customers acquired through **inbound** tend to churn most often.
- Tenants acquired through **referral** also show the highest growth potential.
- In addition to referral, **outbound** and **trial** tenants also have high stay-and-grow scores.

### Visual Output
- `results/acquisition_channel_stay_grow.png`

## 5. Dashboard

The brief asked for a live or shareable, self-explanatory, and actionable dashboard. 

The dashboard source is in `dashboard/streamlit_app.py`, and the dashboard is published on Streamlit Community Cloud.

Deployed dashboard:
- `https://syskit.streamlit.app/`

### What the Dashboard Answers
- Health and renewal distribution across the base
- 12-week usage trend by health segment
- Customer Success queues for at-risk and quiet accounts
- Expansion candidate list for Sales and Customer Success

## 6. Executive Summary

The brief requested a short non-technical summary for the VP of Customer Success and the VP of Sales. It is included in:
- `docs/executive_summary.pdf`
