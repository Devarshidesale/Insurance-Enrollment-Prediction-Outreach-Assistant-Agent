# Technical Report: Insurance Enrollment Prediction & Outreach Assistant Agent

## Executive Summary

This deliverable defines a production-oriented architecture for predicting voluntary insurance enrollment and supporting regional outreach prioritization under operational and compliance constraints. The design emphasizes data quality resilience, strict leakage controls, demographic fairness safeguards, and agent-level refusal policies for safe decision support.

---

## 1) Data Audit & Cleaning Decisions

### 1.1 Source Data Characteristics
The workflow integrates:
- Employee-level raw data (`employees_raw.csv`)
- Region-level benefit and operational profile data (`region_benefit_profiles.csv`)

The raw employee feed is expected to be noisy and non-standardized due to upstream HRIS inconsistencies, manual entry, and asynchronous updates.

### 1.2 Duplicate Record Resolution for `employee_id`
Duplicate rows with identical `employee_id` but conflicting values can arise from retroactive corrections, ingestion retries, or system merges. A deterministic policy is required to avoid unstable labels and training leakage.

**Resolution policy:**
1. Group rows by `employee_id`.
2. Prefer the row with the latest trustworthy timestamp (`application_date` if valid, else `last_contact_date`).
3. If timestamps are tied or unavailable, apply a data completeness score (fewer nulls wins).
4. If `enrolled` labels still conflict after tie-breaking, mark the record as a conflict case in audit output and exclude from model training unless a business-approved adjudication rule is available.

This policy preserves reproducibility and avoids label contamination.

### 1.3 Date Normalization Strategy
Fields such as `application_date` and `last_contact_date` may contain mixed formats (ISO, US-style, textual month names, or malformed values).

**Normalization approach:**
- Parse with strict ordered format attempts (e.g., ISO first), then fallback parsing.
- Convert to canonical UTC-compatible date representation for downstream consistency.
- Capture parse failures in a quality report.
- Avoid silent coercion for impossible dates; set null and flag for monitoring.

A separate anomaly counter should track date parsing failure rates by region to identify localized upstream quality issues.

### 1.4 Categorical Cleanup
Operational categorical noise includes inconsistent channel strings: `Email`, `EMAIL`, `e-mail`, `email `, etc.

**Cleanup policy:**
- Apply trimming, lowercasing, punctuation normalization.
- Map known aliases to canonical values (e.g., `email`, `phone`, `sms`, `in_person`).
- Route unmatched categories to an explicit `other_unknown` bucket.

Canonical categoricals reduce feature sparsity and improve interpretability.

### 1.5 Sentinel Code Handling: `prior_year_enrolled = 1`
A critical semantic caveat: `prior_year_enrolled = 1` denotes **“New Hire / No Prior Record”**, not a true positive prior enrollment indicator.

**Required handling:**
- Reinterpret this sentinel into a dedicated state (e.g., `new_hire_no_history`).
- Do not treat this value as equivalent to historical enrollment behavior.
- Include a quality assertion preventing accidental binary reinterpretation.

This prevents model distortion from semantic inversion.

### 1.6 Join Integrity: `employees_raw` ↔ `region_benefit_profiles`
The integration key is `region`.

**Join controls:**
- Pre-join cardinality check for unique region keys in profile data.
- Post-join coverage checks:
  - Employee rows with missing matched region profile
  - Region profiles never referenced by any employee
- Hard-fail training pipeline if join coverage drops below agreed threshold.

A join exception report should be retained for governance reviews.

---

## 2) Feature Taxonomy & Compliance Checkpoint

### 2.1 Feature Classes
Feature definitions should be classified into:
1. **Allowed operational features** (interaction history, plan context, region service characteristics)
2. **Restricted leakage features** (direct or proxy target reconstruction)
3. **Protected demographic attributes** (fairness-sensitive fields)

### 2.2 Forbidden Target Leakage
The following are disallowed in modeling and ranking:
- `legacy_propensity_score` (known target reconstruction risk)
- `hist_enrollment_rate_region` and similar regional target aggregates derived from outcome prevalence

These variables either directly encode target behavior or inject post-hoc target statistics that inflate offline metrics and degrade production reliability.

### 2.3 Demographic Compliance Exclusions
The following fields must be excluded from model inputs:
- `gender`
- `marital_status`
- `age`

Rationale:
- Reduces disparate treatment risk in outreach prioritization
- Aligns with non-discriminatory communication policies
- Prevents explanation narratives from exposing protected reasoning paths

### 2.4 Compliance Checkpoint Procedure
A mandatory pre-training checkpoint should:
1. Verify blocked columns are absent from training matrix
2. Validate engineered features for proxy leakage patterns where feasible
3. Generate an immutable compliance artifact with pass/fail status and timestamp

No model training should proceed if the checkpoint fails.

---

## 3) Model Evaluation Strategy & Metrics

### 3.1 Baseline Comparators
Two baselines establish minimum acceptable performance:

1. **Majority Class Baseline**
   - Predict all outcomes as the dominant class
   - Provides calibration and discrimination floor

2. **Rule Heuristic: `has_dependents == Yes`**
   - Simple business prior used as a naïve ranking signal
   - Useful to demonstrate incremental value from learned models

### 3.2 Core Statistical Metrics
- **AUC-ROC:** rank discrimination across thresholds
- **Log-Loss:** probabilistic calibration quality

AUC-ROC alone is insufficient for outreach operations; calibration and business-selective ranking matter.

### 3.3 Business Metrics Under Capacity Constraints
Regional outreach is capped by `hr_outreach_capacity`. Therefore, ranking quality at operational cutoffs is critical.

- **Precision@K:** fraction of true enrollees among top-K ranked contacts
- **Lift@K:** enrichment over population baseline at top-K

Where K is region-specific and bounded by capacity. Reporting should include both macro and per-region slices.

### 3.4 Validation Design Notes
- Prefer temporal or grouped validation if campaign timing may induce leakage.
- Track variance of Precision@K/Lift@K across regions to detect instability.
- Retain threshold-independent and threshold-dependent views for governance and planning.

---

## 4) Agent Architecture & Refusal Logic

### 4.1 Tooling Interface Contract
The outreach assistant should expose the following tool-level interfaces:

```text
predict_enrollment(employee_record: dict) -> dict
rank_outreach_candidates(candidates: list[dict], region: str, k: int) -> list[dict]
lookup_region_profile(region: str) -> dict
explain_prediction(employee_record: dict, prediction_context: dict) -> str
```

These signatures define expected I/O behavior while leaving implementation language and backend details flexible.

### 4.2 Router Responsibilities
`agent_router` should:
1. Validate user intent and tool eligibility
2. Enforce guardrails before dispatch
3. Attach compliance metadata to responses
4. Return refusal payloads when policy violations are detected

### 4.3 Mandatory Refusal Rule: Target Leakage
Any query attempting to request, include, or optimize using `legacy_propensity_score` must be refused.

**Refusal triggers include:**
- Direct request to use `legacy_propensity_score`
- Prompted feature override to include blocked features
- Indirect attempts to infer or back-calculate prohibited score usage

**Refusal response characteristics:**
- Clear policy-based reason
- Safe alternative suggestion (allowed features only)
- No partial leakage-compliant workaround that still references blocked signal

### 4.4 Mandatory Explanation Guardrail: Non-Demographic Output
`explain_prediction` outputs must never cite protected demographics:
- `gender`
- `marital_status`
- `age`

A post-generation filter should detect and redact prohibited demographic tokens and close semantic variants before returning text.

### 4.5 Agent Output Governance
All agent outputs should include:
- Decision trace identifier
- Guardrail status (pass/refused)
- Feature policy version

This supports auditability and incident response in enterprise settings.

---

## 5) Limitations & Future Improvements

### 5.1 Current Limitations
1. **Data quality dependency:** severe upstream schema drift can reduce pipeline reliability.
2. **Label uncertainty:** unresolved duplicate-label conflicts reduce effective training set size.
3. **Region heterogeneity:** behavior differs by local plan design and communication norms.
4. **Static training assumptions:** model may degrade as campaign behavior and benefit policies evolve.

### 5.2 Model Drift and Monitoring Needs
Future productionization should include:
- Population stability tracking for key non-protected features
- Performance drift monitoring (AUC, calibration, Precision@K by region)
- Alerting thresholds and retraining triggers

### 5.3 Future Agent Capabilities
Planned enhancements:
- Counterfactual outreach recommendations constrained to compliant features
- Human-in-the-loop adjudication for borderline rankings
- Region-level simulation for capacity and expected conversion trade-offs
- Formal policy engine integration for dynamic guardrail updates

### 5.4 Governance Expansion
A stronger governance posture should eventually include:
- Periodic fairness audits with legal/compliance oversight
- Signed model cards and dataset cards per release
- End-to-end lineage from raw extract to ranked recommendation set

---

## Conclusion

The proposed architecture balances predictive utility with operational realism and compliance control. By combining strict data cleaning policies, explicit feature governance, business-aligned metrics, and enforceable agent guardrails, the system is positioned to support reliable and responsible insurance outreach prioritization at enterprise scale.
