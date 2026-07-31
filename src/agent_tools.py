"""
agent_tools.py
==============
Stateless tool functions consumed by the enrollment outreach agent.

Design principles
-----------------
- Tools load data / model lazily at first call so no imports are needed at
  module level in the router.
- Configuration (paths, column names, forbidden features) is read from a
  central ModelConfig object – change the config and nothing else breaks.
- Tools are plain Python functions that return dicts or DataFrames; the router
  is responsible for formatting output for the user.
"""

from __future__ import annotations

import os
import json
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CENTRAL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Single place to update paths & column names if data or model changes."""

    # ── Paths ────────────────────────────────────────────────────────────────
    base_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1]
    )

    @property
    def data_processed_dir(self) -> Path:
        return self.base_dir / "data" / "processed"

    @property
    def data_raw_dir(self) -> Path:
        return self.base_dir / "data" / "raw_data"

    @property
    def model_dir(self) -> Path:
        return self.base_dir / "models"

    @property
    def employees_csv(self) -> Path:
        return self.data_processed_dir / "employees_processed_no_leaky_features.csv"

    @property
    def region_processed_csv(self) -> Path:
        return self.data_processed_dir / "region_processed.csv"

    @property
    def region_profiles_csv(self) -> Path:
        return self.data_raw_dir / "region_benefit_profiles.csv"

    @property
    def model_path(self) -> Path:
        return self.model_dir / "xgb_noleak_tuned.joblib"

    @property
    def feature_list_path(self) -> Path:
        return self.model_dir / "feature_columns.json"

    # ── Feature column lists ──────────────────────────────────────────────────
    categorical_cols: list[str] = field(default_factory=lambda: [
        "gender",
        "marital_status",
        "employment_type",
        "region",
        "has_dependents",
        "broker_channel",
    ])

    # ── Forbidden features (leaky / derived from target) ─────────────────────
    forbidden_cols: list[str] = field(default_factory=lambda: [
        "legacy_propensity_score",
        "hist_enrollment_rate_region",
        "contact_to_application_days",
        "days_contact_to_app",
        "is_contact_after_app",
        "days_since_last_contact",
        "has_applied",
        "app_month",
        "app_day_of_week",
        "application_date",
        "has_missing_app_date",
        "days_since_application",
        "month_of_application",
        "day_of_week_application",
        "plan_tier_requested",
    ])

    # ── Explanation: columns NOT to cite for fairness/legal reasons ───────────
    explanation_blocked_cols: list[str] = field(default_factory=lambda: [
        "gender",
        "marital_status",
        "age",
    ])

    # ── Human-readable labels for features used in explanations ──────────────
    feature_labels: dict[str, str] = field(default_factory=lambda: {
        "has_dependents":            "has dependents",
        "employment_type":           "employment type",
        "salary_band":               "salary band",
        "salary_clean":              "salary",
        "is_new_hire":               "is a new hire",
        "prior_year_enrolled_clean": "prior-year enrollment status",
        "salary_age_ratio":          "salary-to-age ratio",
        "salary_per_tenure":         "salary-per-tenure",
        "n_employees_region":        "region size",
        "tenure_years":              "tenure",
        "broker_channel":            "broker channel",
        "state_mandate_level_clean": "state mandate level",
        "avg_premium_cost_usd":      "average premium cost",
        "benefits_broker_rating":    "benefits broker rating",
        "open_enrollment_window_days": "open enrollment window length",
        "avg_salary_region":         "regional average salary",
        "hr_outreach_capacity":      "HR outreach capacity",
        "has_outreach_note":         "previous outreach note",
    })

    # ── Score threshold above which we say "likely to enroll" ────────────────
    enroll_threshold: float = 0.5


# Singleton config – import this in router and tools
CONFIG = ModelConfig()


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LAZY-LOADED SHARED STATE (model + data)
# ─────────────────────────────────────────────────────────────────────────────

class _ModelState:
    """Cache for the trained model, encoders, and processed data."""

    def __init__(self):
        self._model = None
        self._encoders: dict[str, LabelEncoder] | None = None
        self._feature_cols: list[str] | None = None
        self._employees: pd.DataFrame | None = None
        self._region_processed: pd.DataFrame | None = None
        self._config = CONFIG

    # ── Model + features ─────────────────────────────────────────────────────
    def _load_model(self):
        """Train and cache the XGBoost model if no saved model is found."""
        import joblib

        cfg = self._config

        if cfg.model_path.exists() and cfg.feature_list_path.exists():
            self._model = joblib.load(cfg.model_path)
            with open(cfg.feature_list_path, "r") as f:
                self._feature_cols = json.load(f)
            # Rebuild encoders from employee data
            self._encoders = _fit_encoders(self.employees, cfg.categorical_cols)
        else:
            # Train inline (fallback so agent always works even without a saved model)
            self._model, self._encoders, self._feature_cols = _train_model_inline(cfg)

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def encoders(self) -> dict[str, LabelEncoder]:
        if self._encoders is None:
            self._load_model()
        return self._encoders

    @property
    def feature_cols(self) -> list[str]:
        if self._feature_cols is None:
            self._load_model()
        return self._feature_cols

    # ── Data ─────────────────────────────────────────────────────────────────
    @property
    def employees(self) -> pd.DataFrame:
        if self._employees is None:
            self._employees = pd.read_csv(self._config.employees_csv)
            self._employees["region"] = (
                self._employees["region"].astype(str).str.strip().str.title()
            )
        return self._employees

    @property
    def region_processed(self) -> pd.DataFrame:
        if self._region_processed is None:
            r = pd.read_csv(self._config.region_processed_csv)
            r["region"] = r["region"].astype(str).str.strip().str.title()
            self._region_processed = r
        return self._region_processed


_STATE = _ModelState()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _fit_encoders(df: pd.DataFrame, cat_cols: list[str]) -> dict[str, LabelEncoder]:
    encoders = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            le.fit(df[col].fillna("Missing").astype(str))
            encoders[col] = le
    return encoders


def _encode_row(row_dict: dict[str, Any], encoders: dict[str, LabelEncoder],
                cat_cols: list[str]) -> dict[str, Any]:
    """Label-encode categorical values in a single record dict."""
    out = dict(row_dict)
    for col in cat_cols:
        if col in out:
            val = str(out[col]) if out[col] is not None else "Missing"
            le = encoders.get(col)
            if le is not None:
                if val not in le.classes_:
                    val = "Missing"  # unseen category → sentinel
                out[col] = int(le.transform([val])[0])
    return out


def _build_feature_vector(row_dict: dict[str, Any], feature_cols: list[str],
                          encoders: dict[str, LabelEncoder],
                          cat_cols: list[str]) -> pd.DataFrame:
    """
    Build a single-row DataFrame ready for model.predict_proba().
    Missing numeric columns are filled with 0 (or median if available).
    """
    encoded = _encode_row(row_dict, encoders, cat_cols)
    vector = {}
    for col in feature_cols:
        vector[col] = encoded.get(col, 0)
    return pd.DataFrame([vector])


def _train_model_inline(cfg: ModelConfig):
    """Train the XGBoost model from scratch (used when no saved model exists)."""
    import xgboost as xgb
    import joblib
    from sklearn.model_selection import train_test_split

    print("[agent_tools] No saved model found – training inline …")

    employees_df = pd.read_csv(cfg.employees_csv)
    region_df = pd.read_csv(cfg.region_processed_csv)
    employees_df["region"] = employees_df["region"].astype(str).str.strip().str.title()
    region_df["region"] = region_df["region"].astype(str).str.strip().str.title()

    df = employees_df.merge(region_df, on="region", how="left", validate="many_to_one")

    # Feature sets (must match train_model.ipynb STEP 2)
    LEAKY_COLS = [
        "legacy_propensity_score",
        "hist_enrollment_rate_region",
    ]
    APPLICATION_LEAKY_COLS = [
        "application_date", "has_applied", "has_missing_app_date",
        "days_since_application", "contact_to_application_days",
        "days_contact_to_app", "month_of_application", "day_of_week_application",
        "app_month", "app_day_of_week", "is_contact_after_app", "plan_tier_requested",
    ]
    ANALYSIS_ONLY = [
        "employee_id", "enrolled", "salary", "prior_year_enrolled",
        "last_contact_date", "last_contact_channel", "outreach_notes",
        "min_salary_allowed", "is_implausible_salary",
        "legacy_propensity_score_missing",
    ]
    ALL_FORBIDDEN = set(LEAKY_COLS) | set(APPLICATION_LEAKY_COLS) | set(ANALYSIS_ONLY)

    cat_cols = cfg.categorical_cols
    usable_raw = set(df.columns) - ALL_FORBIDDEN
    numeric_cols = [c for c in usable_raw if c not in cat_cols]
    feature_cols = numeric_cols + cat_cols

    # Encode
    encoders = _fit_encoders(df, cat_cols)
    df_enc = df.copy()
    for col in cat_cols:
        if col in df_enc.columns:
            df_enc[col] = df_enc[col].fillna("Missing").astype(str)
            df_enc[col] = encoders[col].transform(df_enc[col])

    X = df_enc[feature_cols].copy()
    y = df["enrolled"].copy()

    for col in numeric_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].fillna(X[col].median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Save artifacts for next run
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, cfg.model_path)
    with open(cfg.feature_list_path, "w") as f:
        json.dump(feature_cols, f)

    print(f"[agent_tools] Model saved → {cfg.model_path}")
    return model, encoders, feature_cols


def _merged_employee_row(employee_id: int) -> dict[str, Any] | None:
    """Return a raw dict for one employee, merged with region features."""
    emp = _STATE.employees
    row = emp[emp["employee_id"] == employee_id]
    if row.empty:
        return None
    row_dict = row.iloc[0].to_dict()

    # Merge in region numeric features
    region_name = str(row_dict.get("region", "")).strip().title()
    rdf = _STATE.region_processed
    r_row = rdf[rdf["region"].str.strip().str.title() == region_name]
    if not r_row.empty:
        for col, val in r_row.iloc[0].to_dict().items():
            if col not in row_dict:
                row_dict[col] = val

    return row_dict


# ─────────────────────────────────────────────────────────────────────────────
# 3.  GUARD: forbidden-feature detector
# ─────────────────────────────────────────────────────────────────────────────

def _check_forbidden_features(row_dict: dict[str, Any]) -> list[str]:
    """Return any forbidden (leaky) column names present in the dict."""
    return [k for k in row_dict if k in CONFIG.forbidden_cols]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TOOL: predict_enrollment
# ─────────────────────────────────────────────────────────────────────────────

def predict_enrollment(employee_id: int | None = None,
                       raw_row: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Run the trained model on cleaned features for one employee.

    Parameters
    ----------
    employee_id : int, optional
        Look up the employee from the processed dataset.
    raw_row : dict, optional
        A raw feature dict (e.g., from a form).  Any forbidden/leaky columns
        are stripped BEFORE prediction with an explicit warning.

    Returns
    -------
    dict with keys:
        employee_id, enrollment_probability, prediction, warnings
    """
    # ── Resolve record ────────────────────────────────────────────────────────
    if employee_id is not None:
        row_dict = _merged_employee_row(employee_id)
        if row_dict is None:
            return {"error": f"Employee {employee_id} not found in dataset."}
    elif raw_row is not None:
        row_dict = dict(raw_row)
        employee_id = row_dict.get("employee_id", "unknown")
    else:
        return {"error": "Provide either employee_id or raw_row."}

    # ── Forbidden-feature guard ───────────────────────────────────────────────
    refusal_notes: list[str] = []
    forbidden_found = _check_forbidden_features(row_dict)
    if forbidden_found:
        refusal_notes.append(
            f"[REFUSAL] The following leaky/forbidden fields were DETECTED and "
            f"EXCLUDED from the prediction (they reconstruct the target or use "
            f"future information): {forbidden_found}. "
            "Using them would produce misleadingly optimistic results."
        )
        for col in forbidden_found:
            row_dict.pop(col, None)

    # ── Build feature vector & predict ───────────────────────────────────────
    X = _build_feature_vector(
        row_dict, _STATE.feature_cols, _STATE.encoders, CONFIG.categorical_cols
    )
    prob = float(_STATE.model.predict_proba(X)[0, 1])
    prediction = "ENROLL" if prob >= CONFIG.enroll_threshold else "NOT ENROLL"

    return {
        "employee_id":            employee_id,
        "enrollment_probability": round(prob, 4),
        "prediction":             prediction,
        "threshold":              CONFIG.enroll_threshold,
        "warnings":               refusal_notes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  TOOL: rank_outreach_candidates
# ─────────────────────────────────────────────────────────────────────────────

def rank_outreach_candidates(
    region: str | None = None,
    top_k: int | None = None,
    exclude_already_enrolled: bool = True,
) -> pd.DataFrame:
    """
    Given a region (and its hr_outreach_capacity), return the top-K
    prioritized employee list sorted by enrollment probability (descending).

    Parameters
    ----------
    region : str, optional
        Region name (case-insensitive). If None, returns across all regions,
        respecting each region's hr_outreach_capacity.
    top_k : int, optional
        Override the region's hr_outreach_capacity.
    exclude_already_enrolled : bool
        If True (default), exclude employees who enrolled last year
        (prior_year_enrolled_clean == 1) since they may re-enroll automatically.

    Returns
    -------
    DataFrame with columns: employee_id, region, enrollment_probability,
                             rank, employment_type, has_dependents, ...
    """
    emp = _STATE.employees.copy()
    rdf = _STATE.region_processed.copy()

    # ── Filter region ─────────────────────────────────────────────────────────
    if region is not None:
        region_title = region.strip().title()
        emp = emp[emp["region"].str.title() == region_title]
        if emp.empty:
            return pd.DataFrame({"error": [f"Region '{region}' not found."]})

        r_row = rdf[rdf["region"].str.title() == region_title]
        capacity = int(r_row["hr_outreach_capacity"].iloc[0]) if not r_row.empty else 50
        k = top_k if top_k is not None else capacity
    else:
        k = top_k  # None → each region uses own capacity

    # ── Optional: skip last-year-enrolled ────────────────────────────────────
    if exclude_already_enrolled and "prior_year_enrolled_clean" in emp.columns:
        emp = emp[emp["prior_year_enrolled_clean"] != 1]

    # ── Merge region features ─────────────────────────────────────────────────
    merged = emp.merge(rdf, on="region", how="left", validate="many_to_one",
                       suffixes=("", "_reg"))

    # ── Predict probabilities for all rows ───────────────────────────────────
    feature_cols = _STATE.feature_cols
    cat_cols = CONFIG.categorical_cols
    encoders = _STATE.encoders

    df_enc = merged.copy()
    for col in cat_cols:
        if col in df_enc.columns:
            le = encoders.get(col)
            if le is None:
                continue
            df_enc[col] = df_enc[col].fillna("Missing").astype(str)
            # Handle unseen categories
            df_enc[col] = df_enc[col].apply(
                lambda v: v if v in le.classes_ else "Missing"
            )
            df_enc[col] = le.transform(df_enc[col])

    available = [c for c in feature_cols if c in df_enc.columns]
    X_all = df_enc[available].copy()
    for col in available:
        if col not in cat_cols:
            X_all[col] = pd.to_numeric(X_all[col], errors="coerce")
            X_all[col] = X_all[col].fillna(X_all[col].median())

    # Fill any missing feature columns with 0
    for col in feature_cols:
        if col not in X_all.columns:
            X_all[col] = 0
    X_all = X_all[feature_cols]

    probs = _STATE.model.predict_proba(X_all)[:, 1]
    merged = merged.copy()
    merged["enrollment_probability"] = np.round(probs, 4)

    # ── Rank ─────────────────────────────────────────────────────────────────
    if region is not None:
        ranked = merged.sort_values("enrollment_probability", ascending=False)
        ranked = ranked.head(k)
        ranked["outreach_rank"] = range(1, len(ranked) + 1)
    else:
        # Per-region ranking respecting each region's capacity
        pieces = []
        for rgn, grp in merged.groupby("region"):
            r_row = rdf[rdf["region"].str.title() == rgn.strip().title()]
            cap = int(r_row["hr_outreach_capacity"].iloc[0]) if not r_row.empty else 50
            cap = top_k if top_k is not None else cap
            g_sorted = grp.sort_values("enrollment_probability", ascending=False).head(cap)
            g_sorted = g_sorted.copy()
            g_sorted["outreach_rank"] = range(1, len(g_sorted) + 1)
            pieces.append(g_sorted)
        ranked = pd.concat(pieces).sort_values(
            ["region", "outreach_rank"], ascending=[True, True]
        )

    # ── Return readable subset ────────────────────────────────────────────────
    keep_cols = [
        "employee_id", "region", "enrollment_probability", "outreach_rank",
        "employment_type", "has_dependents", "salary_band",
        "prior_year_enrolled_clean", "is_new_hire", "tenure_years",
    ]
    keep_cols = [c for c in keep_cols if c in ranked.columns]
    return ranked[keep_cols].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  TOOL: lookup_region_profile
# ─────────────────────────────────────────────────────────────────────────────

def lookup_region_profile(region: str) -> dict[str, Any]:
    """
    Return region-level stats from region_benefit_profiles.csv.

    Parameters
    ----------
    region : str  e.g. "Midwest", "south"

    Returns
    -------
    dict of region statistics, or an error dict.
    """
    profiles = pd.read_csv(CONFIG.region_profiles_csv)
    profiles["region"] = profiles["region"].astype(str).str.strip().str.title()
    match = profiles[profiles["region"] == region.strip().title()]

    if match.empty:
        available = profiles["region"].tolist()
        return {
            "error": f"Region '{region}' not found. Available: {available}"
        }

    row = match.iloc[0].to_dict()

    # Enrich with outreach capacity utilisation if possible
    emp = _STATE.employees
    region_emp = emp[emp["region"].str.title() == region.strip().title()]
    row["total_employees_in_data"] = len(region_emp)
    row["current_enrolled_rate"] = round(
        region_emp["enrolled"].mean(), 4
    ) if "enrolled" in region_emp.columns and not region_emp.empty else None

    return row


# ─────────────────────────────────────────────────────────────────────────────
# 7.  TOOL: explain_prediction
# ─────────────────────────────────────────────────────────────────────────────

def explain_prediction(employee_id: int) -> dict[str, Any]:
    """
    Generate a short natural-language explanation for one prediction.

    Fairness guardrails
    -------------------
    The explanation will NOT cite gender, marital_status, or age even if those
    features appear in the model.  Doing so could constitute adverse decision
    language that is legally risky and ethically inappropriate.

    Returns
    -------
    dict with keys: employee_id, prediction, enrollment_probability,
                    explanation, fairness_note
    """
    # ── Get prediction ────────────────────────────────────────────────────────
    pred_result = predict_enrollment(employee_id=employee_id)
    if "error" in pred_result:
        return pred_result

    prob = pred_result["enrollment_probability"]
    prediction = pred_result["prediction"]

    # ── Get feature importances from model ────────────────────────────────────
    model = _STATE.model
    feature_cols = _STATE.feature_cols
    importances = dict(zip(feature_cols, model.feature_importances_))

    # ── Get this employee's feature values ────────────────────────────────────
    row_dict = _merged_employee_row(employee_id)
    if row_dict is None:
        return {"error": f"Employee {employee_id} not found."}

    # Encode for comparison
    encoded = _encode_row(row_dict, _STATE.encoders, CONFIG.categorical_cols)

    # ── Identify top contributing features ───────────────────────────────────
    BLOCKED = set(CONFIG.explanation_blocked_cols)  # gender, marital_status, age

    scored_features: list[tuple[str, float, Any]] = []
    for col in feature_cols:
        if col in BLOCKED:
            continue   # silently skip — fairness guardrail
        importance = importances.get(col, 0.0)
        raw_val = row_dict.get(col)
        if importance > 0.01 and raw_val is not None:
            scored_features.append((col, importance, raw_val))

    scored_features.sort(key=lambda x: x[1], reverse=True)
    top_features = scored_features[:5]

    # ── Build natural-language bullets ───────────────────────────────────────
    label_map = CONFIG.feature_labels
    bullets: list[str] = []

    for col, imp, val in top_features:
        label = label_map.get(col, col.replace("_", " "))

        # Format value
        if col == "has_dependents":
            val_str = str(val)
        elif col == "is_new_hire":
            val_str = "Yes" if val == 1 else "No"
        elif col == "prior_year_enrolled_clean":
            if val == 1:
                val_str = "enrolled last year"
            elif val == 0:
                val_str = "not enrolled last year"
            else:
                val_str = "new hire (no prior record)"
        elif col == "employment_type":
            val_str = str(val)
        elif col in ("salary_clean", "avg_salary_region", "avg_premium_cost_usd"):
            val_str = f"${float(val):,.0f}"
        elif col in ("salary_per_tenure", "salary_age_ratio"):
            val_str = f"{float(val):,.1f}"
        elif col == "salary_band":
            bands = {0: "Low", 1: "Lower-Mid", 2: "Mid", 3: "Upper-Mid", 4: "High"}
            val_str = bands.get(int(val), str(val))
        elif col == "benefits_broker_rating":
            val_str = f"{float(val):.1f}/5.0"
        elif col == "tenure_years":
            val_str = f"{float(val):.1f} years"
        elif col == "state_mandate_level_clean":
            levels = {0: "Low", 1: "Medium", 2: "High"}
            val_str = levels.get(int(val), str(val))
        elif col == "broker_channel":
            val_str = str(val)
        else:
            try:
                val_str = f"{float(val):.2f}"
            except (TypeError, ValueError):
                val_str = str(val)

        bullets.append(f"• {label.capitalize()}: {val_str} (feature weight: {imp:.3f})")

    direction = "likely" if prediction == "ENROLL" else "unlikely"
    summary = (
        f"Employee {employee_id} is {direction} to enroll "
        f"(probability: {prob:.1%}). "
        f"The top contributing factors are:"
    )

    explanation = summary + "\n" + "\n".join(bullets)

    fairness_note = (
        "NOTE: This explanation deliberately omits demographic attributes "
        "(gender, marital status, age) to avoid legally or ethically "
        "sensitive language in outreach decisions, even if those features "
        "were included in the model."
    )

    return {
        "employee_id":            employee_id,
        "prediction":             prediction,
        "enrollment_probability": prob,
        "explanation":            explanation,
        "fairness_note":          fairness_note,
        "warnings":               pred_result.get("warnings", []),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8.  TOOL: validate_raw_row
# ─────────────────────────────────────────────────────────────────────────────

def validate_raw_row(raw_row: dict[str, Any]) -> dict[str, Any]:
    """
    Inspect a raw employee record dict and report data quality issues:
      - Forbidden / leaky columns present
      - Missing required feature columns
      - Implausible salary values
      - Unknown categorical values

    Returns
    -------
    dict with keys: issues (list[str]), warnings (list[str]), clean_row (dict)
    """
    issues: list[str] = []
    warnings_out: list[str] = []
    clean = dict(raw_row)

    # ── Forbidden features ────────────────────────────────────────────────────
    forbidden_found = _check_forbidden_features(clean)
    if forbidden_found:
        issues.append(
            f"[LEAKAGE] Forbidden columns detected and removed: {forbidden_found}. "
            "These reconstruct the target or use future information."
        )
        for col in forbidden_found:
            clean.pop(col, None)

    # ── Required columns missing ──────────────────────────────────────────────
    required = [c for c in _STATE.feature_cols
                if c not in CONFIG.categorical_cols]
    missing_required = [c for c in required if c not in clean or pd.isnull(clean.get(c))]
    if missing_required:
        issues.append(f"[MISSING] Required numeric features not provided: {missing_required}")

    # ── Salary plausibility ───────────────────────────────────────────────────
    salary = clean.get("salary_clean") or clean.get("salary")
    if salary is not None:
        try:
            s = float(salary)
            if s < 0:
                issues.append(f"[INVALID] salary value is negative: {s}")
            elif s < 15_000:
                warnings_out.append(f"[WARN] salary value {s} is very low (< $15,000).")
            elif s > 500_000:
                warnings_out.append(f"[WARN] salary value {s} is very high (> $500,000).")
        except (TypeError, ValueError):
            issues.append(f"[INVALID] salary value cannot be parsed: {salary}")

    # ── Categorical validity ──────────────────────────────────────────────────
    encoders = _STATE.encoders
    for col in CONFIG.categorical_cols:
        if col in clean and col in encoders:
            val = str(clean[col]) if clean[col] is not None else "Missing"
            if val not in encoders[col].classes_:
                warnings_out.append(
                    f"[WARN] Unknown category '{val}' for '{col}'. "
                    f"Known values: {list(encoders[col].classes_)}"
                )

    status = "PASS" if not issues else "FAIL"
    return {
        "status":   status,
        "issues":   issues,
        "warnings": warnings_out,
        "clean_row": clean,
    }
