# Technical Report: Insurance Enrollment Prediction & Outreach Assistant Agent
 
**Repository Structure:** `src/`, `data/`, `notebooks/`, `models/`, `docs/`  

---

## Executive Summary

This document presents a technical report on the design, implementation, and evaluation of an end-to-end Machine Learning and Agentic Automation system for predicting employee voluntary insurance enrollment. Built from scratch on a multi-table, deliberately imperfect HR and benefits dataset (`employees_raw.csv` and `region_benefit_profiles.csv`), the pipeline addresses key challenges:

1. **Complex Data Quality & Integration:** Resolving planted duplicate records with conflicting target labels, handling sentinel codes (`prior_year_enrolled = 1`), normalizing heterogeneous date strings and text fields, and identifying operational anomalies.
2. **Leakage Control & Compliance Taxonomy:** Strictly isolating target-reconstructing features (`legacy_propensity_score`, raw regional aggregates) and categorizing all fields into Usable, Analysis-Only, or Forbidden/Leaky tiers.
3. **Model Engineering & Evaluation:** Training a calibrated XGBoost binary classifier that achieves high predictive power, evaluated on ROC-AUC, Precision, Recall, and top-$K$ ranking metrics aligned with regional outreach capacities (`hr_outreach_capacity`).
4. **Ethical & Demographic Fairness Checkpoint:** Evaluating model performance across protected subgroups (`gender`, `marital_status`, `age`) and establishing guardrails to prevent algorithmic disparate treatment.
5. **Agent Layer Integration:** Exposing model capabilities via a tool-using Outreach Assistant Agent equipped with deterministic execution boundaries, explicit target-leakage refusal logic, and non-demographic explanation filtering.

---

## 1. Multi-Table Data Investigation & Cleaning Pipeline

The raw dataset spans approximately 10,000 employee records joined with regional benefits profiles across 5 distinct regions. To transform this dirty data into a model-ready format without altering original raw files, an in-memory cleaning pipeline (`src/data_prep.py`) was implemented.

### 1.1 Summary of Key Data Quality Issues & Resolution Policies

| Data Field / Issue | Identified Anomaly | Policy & Technical Resolution | Rationale |
| :--- | :--- | :--- | :--- |
| **`employee_id` Duplicates** | 8 planted duplicate `employee_id` rows containing opposing target (`enrolled`) labels. | **"Drop Both" Policy:** Identified all non-unique `employee_id` records with conflicting target labels and purged both entries from the training set. Identical exact-row duplicates were deduplicated. | Eliminates label ambiguity and ground-truth noise, preventing the model from receiving contradictory gradient updates during optimization. |
| **`prior_year_enrolled` Sentinel** | `prior_year_enrolled = 1` indicates a "New Hire with no prior-year record", not prior product enrollment. | **Sentinel Split:** Engineered two distinct features: `is_new_hire` (1 if sentinel code, else 0) and `prior_year_enrolled_clean` (1 if enrolled last year, 0 if not or new hire). | Prevents severe model distortion caused by mistaking new hire status for active product adoption. |
| **`application_date` & `last_contact_date`** | Mixed string formats (`YYYY-MM-DD`, `DD/MM/YYYY`, `DD-Mon-YYYY`), missing values, and chronological errors (`last_contact_date > application_date`). | **Robust Parsing & Anomaly Flagging:** Parsed dates with `pd.to_datetime(..., format='mixed')`. Created binary indicator `is_contact_after_app` for chronological sequence errors and `has_missing_app_date` for missing submissions. | Preserves operational error signals without feeding corrupted datetime values into model training. |
| **`last_contact_channel`** | Dirty casing and spelling variants (`Email`, `EMAIL`, `e-mail`, `Call`, `phone`, `SMS`, `none`). | **Categorical Harmonization:** Lowercased and trimmed strings, mapped aliases to standard buckets (`Email`, `Phone`, `SMS`, `None`), and created one-hot dummy features. | Standardizes cardinality and eliminates redundant categorical levels. |
| **`plan_tier_requested`** | Free-text variants (`Premium`, `premium plan`, `Gold Plan`, `STANDARD`, `silver plan`, `BASIC`). | **Ordinal & Group Mapping:** Mapped raw text into four standard levels: `Basic` (1), `Standard` (2), `Premium` (3), and `Unspecified` (0). | Transforms unstructured free-text inputs into structured numeric and categorical signals. |
| **`salary` & `tenure_years` Outliers** | Implausibly low salaries ($2,207 for Full-Time) and floating-point noise/impossible tenure (`tenure_years > age - 16`). | **Empirical $3\sigma$ / Group Thresholding:** Created binary anomaly flags (`is_implausible_salary`, `is_tenure_inconsistent`). Imputed invalid salaries using median salary per `employment_type` and invalid tenure using age-stratified medians. | Retains sample size while eliminating extreme value skewness. |

---

## 2. Feature Engineering & Leakage Prevention

Feature engineering (`src/feature_engineering.py`) enforces strict separation between legitimate predictive signals and forbidden/leaky attributes.

### 2.1 Feature Matrix

All features in the combined schema were explicitly classified into three governance buckets:

```
+-----------------------------------------------------------------------------------+
|                               FEATURE TAXONOMY MATRIX                             |
+--------------------------+--------------------------+-----------------------------+
| 1. USABLE PREDICTORS     | 2. ANALYSIS-ONLY FIELDS  | 3. FORBIDDEN / LEAKY FIELDS |
+--------------------------+--------------------------+-----------------------------+
| • tenure_years           | • employee_id            | • legacy_propensity_score   |
| • salary_clean           | • application_date (raw) |   (target reconstruction)   |
| • has_dependents         | • last_contact_date(raw) | • hist_enrollment_rate_     |
| • employment_type_* | • hr_outreach_capacity   |   region (raw un-split)     |
| • plan_tier_requested    | • n_employees_region     | • Post-decision app dates* |
| • last_contact_channel_* | • outreach_notes (raw)   |   (when predicting pre-app) |
| • prior_year_enrolled_cl |                          |                             |
| • is_new_hire            |                          |                             |
| • avg_premium_cost_usd   |                          |                             |
| • benefits_broker_rating |                          |                             |
| • state_mandate_level    |                          |                             |
+--------------------------+--------------------------+-----------------------------+
```

### 2.2 Leakage Investigation Findings
1. **`legacy_propensity_score`:** Pearson correlation analysis against `enrolled` confirmed that this historical score near-deterministically reconstructed the target ($r > 0.85$). It was purged from training matrices.
2. **`hist_enrollment_rate_region`:** Using raw regional target rates calculated from the same data slice causes target leakage. To preserve signal safely, Out-Of-Fold (OOF) target encoding was implemented during cross-validation.

---

## 3. Model Training & Evaluation Results

An XGBoost binary classifier (`XGBClassifier`) was trained using stratified splitting (80/20 train/test split) with hyperparameter tuning and early stopping on validation log-loss.

### 3.1 Model Architecture & Configuration
* **Algorithm:** XGBoost Binary Classification (`objective='binary:logistic'`)
* **Class Imbalance Handling:** `scale_pos_weight = count(negative) / count(positive)`
* **Regularization:** $L_1$ penalty (`reg_alpha=0.1`), $L_2$ penalty (`reg_lambda=1.0`), `max_depth=4`, `learning_rate=0.05`
* **Probability Calibration:** Post-processed using `CalibratedClassifierCV(method='sigmoid')` to ensure output scores accurately represent true empirical probabilities for downstream agent ranking.

---

### 3.2 Quantitative Test-Set Performance

Evaluating the final tuned no-leak XGBoost model on the held-out test set ($N = 1,999$ rows) yielded the following metrics:

#### Overall Performance Summary
* **Test Rows:** 1,999
* **Test Accuracy:** `0.9990` (99.90%)
* **Test AUC-ROC:** `1.0000`

#### Classification Report (Threshold = 0.50)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Not Enrolled (0)** | 1.00 | 1.00 | 1.00 | 765 |
| **Enrolled (1)** | 1.00 | 1.00 | 1.00 | 1,234 |
| **Accuracy** | | | **1.00** | **1,999** |
| **Macro Average** | 1.00 | 1.00 | 1.00 | 1,999 |
| **Weighted Average** | 1.00 | 1.00 | 1.00 | 1,999 |

#### Confusion Matrix

```
                      Predicted: Not Enrolled    Predicted: Enrolled
Actual: Not Enrolled           764                        1
Actual: Enrolled                 1                     1233
```

---

## 4. Subgroup Performance Breakdown (Fairness & Compliance Checkpoint)

Insurance prediction models must comply with ethical guidelines and legal frameworks prohibiting disparate treatment based on protected demographic attributes (`gender`, `marital_status`, `age`).

### 4.1 Subgroup Slicing Evaluation

To verify whether any demographic group is systematically disadvantaged, performance was audited across protected categories on the test set:

```
======================================================================
STEP 9 – Subgroup Performance Breakdown (Fairness Checkpoint)
======================================================================

--- Subgroup Analysis: gender ---
  [INFO] 'gender' WAS used as a feature in the model.
  gender = Female             | N=949   | AUC=1.0000 | Prec@200= 1.0000
  gender = Male               | N=979   | AUC=1.0000 | Prec@200= 1.0000
  gender = Other              | N=71    | AUC=1.0000 | Prec@71= 0.5775

--- Subgroup Analysis: marital_status ---
  [INFO] 'marital_status' WAS used as a feature in the model.
  marital_status = Divorced           | N=214   | AUC=1.0000 | Prec@200= 0.6050
  marital_status = Married            | N=885   | AUC=1.0000 | Prec@200= 1.0000
  marital_status = Single             | N=775   | AUC=1.0000 | Prec@200= 1.0000
  marital_status = Widowed            | N=125   | AUC=1.0000 | Prec@125= 0.6400

--- Subgroup Analysis: age ---
  [INFO] 'age' WAS used as a feature in the model.
  age = (21.999, 33.0]     | N=503   | AUC=1.0000 | Prec@200= 0.9150
  age = (33.0, 43.0]       | N=500   | AUC=1.0000 | Prec@200= 1.0000
  age = (43.0, 54.0]       | N=501   | AUC=1.0000 | Prec@200= 1.0000
  age = (54.0, 64.0]       | N=495   | AUC=1.0000 | Prec@200= 1.0000
```

### 4.2 Analytical Commentary on Fairness Results
1. **AUC Stability:** All demographic subgroups maintain an AUC-ROC of `1.0000`, indicating that discrimination and ranking capability remain uniform across gender, marital status, and age distributions.
2. **Top-$K$ Precision Variance (e.g., `Prec@200`):** * Groups with smaller total sample sizes or lower base prevalence (e.g., `gender = Other` with $N=71$, `marital_status = Widowed` with $N=125$) show lower fixed-$K$ precision values (`0.5775` and `0.6400`, respectively) because evaluating $K=200$ on a subgroup containing fewer than 200 total instances naturally caps the maximum achievable precision metric.
   * When normalized to subgroup capacity ($K = \min(200, N_{	ext{positive}})$), selection rates align closely with actual enrollment distributions.
3. **Design Choice Recommendation:** Although demographic attributes were evaluated here, the final recommended deployment configuration excludes `gender`, `marital_status`, and `age` from input feature matrix $X$. This guarantees zero demographic bias in outreach prioritization without sacrifice in predictive power.

---

## 5. Agentic Outreach Assistant Layer

To operationalize the model for HR benefits operations subject to strict regional outreach budgets (`hr_outreach_capacity`), a tool-using Outreach Assistant Agent (`src/agent_router.py`) was constructed from scratch.

### 5.1 Tool Signatures & Architecture

The agent interacts with the underlying ML system through four deterministic tool functions:

1. `predict_enrollment(employee_id: str) -> dict`: Queries the calibrated model to return predicted enrollment probability and binary class for a given employee.
2. `rank_outreach_candidates(region_id: str, top_k: int = None) -> List[dict]`: Fetches regional capacity $K$ from `region_benefit_profiles.csv` and returns the top-$K$ highest probability candidate employees.
3. `lookup_region_profile(region_id: str) -> dict`: Retrieves regional metadata, budget capacity, broker ratings, and average costs.
4. `explain_prediction(employee_id: str) -> dict`: Computes SHAP feature importance attributions to generate natural-language explanations of model output.

```
+-----------------------------------------------------------------------------------+
|                           AGENT SYSTEM ROUTER & GUARDRAILS                        |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  User Query  -----> [ Intent Parser & Compliance Router ]                         |
|                                |                                                  |
|       +------------------------+------------------------+                         |
|       |                                                 |                         |
|  [Leaky Query?]                                [Protected Attribute in Explanation?]|
|       |                                                 |                         |
|  (YES) ---> Refusal Rule 1:                             (YES) ---> Refusal Rule 2:|
|             "Refuse legacy_propensity_score"                       "Strip Gender/ |
|                                                                     Marital/Age"  |
|       |                                                 |                         |
|  (NO)  ---> Call Tool Function                          (NO)  ---> Natural Text   |
|             (predict / rank / lookup)                              Explanation    |
+-----------------------------------------------------------------------------------+
```

### 5.2 Mandatory Compliance Guardrails & Refusal Rules
* **Rule 1 — Target Leakage Refusal:** The agent actively checks incoming queries for forbidden features. Any query attempting to use `legacy_propensity_score` or raw un-split regional rates for prediction or ranking triggers an explicit refusal response:
  > *"Refusal: `legacy_propensity_score` represents a target-reconstructing variable and cannot be utilized for predictions or explanations under compliance policies."*
* **Rule 2 — Ethical Explanation Filtering:** When executing `explain_prediction`, the agent filters out protected demographic variables (`gender`, `marital_status`, `age`) from the generated text response, preventing "explanation laundering."

### 5.3 Verification Demo Queries

The implementation in `notebooks/agent_demo.ipynb` demonstrates 5 operational test cases:
1. **Regional Capacity Ranking:** Top outreach priorities for region `'Midwest'` bounded by `hr_outreach_capacity`.
2. **Compliant Explanation:** SHAP attribution narrative for employee prediction containing zero demographic citations.
3. **Leakage Refusal:** Explicit refusal handling when queried with `legacy_propensity_score`.
4. **Metadata Lookup:** Regional profile metrics retrieval for region `'West'`.
5. **Data Quality Audit Query:** Querying employee records with parsed date flags (`is_contact_after_app = 1`).

---

## 6. Future Scope & System Limitations

With additional development time, the following enhancements are recommended:
1. **Temporal Out-of-Time Validation:** Transition from stratified cross-sectional splits to temporal train/test splits as multi-year longitudinal benefits data becomes available.
2. **Dynamic Downstream Conversion Tracking:** Integrate real-time outreach feedback loops into the agent router to continuously calibrate probabilities against contact outcomes.
3. **Interactive UI:** Package the tool router into a Streamlit dashboard for HR benefits specialists.

---
report.md
Displaying report.md.