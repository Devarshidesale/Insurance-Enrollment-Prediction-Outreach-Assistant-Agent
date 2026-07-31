"""
train_and_save.py
=================
Standalone training script (mirrors train_model.ipynb Steps 1-9).
Run this once to save the model artifact; the agent loads it lazily.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PROC = os.path.join(BASE, "..", "data", "processed")
MODEL_DIR = os.path.join(BASE, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

EMP_CSV    = os.path.join(DATA_PROC, "employees_processed_no_leaky_features.csv")
REGION_CSV = os.path.join(DATA_PROC, "region_processed.csv")

print("=" * 70)
print("STEP 1 – Loading & Merging Processed Data")
print("=" * 70)
employees_df = pd.read_csv(EMP_CSV)
region_df    = pd.read_csv(REGION_CSV)
employees_df["region"] = employees_df["region"].astype(str).str.strip().str.title()
region_df["region"]    = region_df["region"].astype(str).str.strip().str.title()
df = employees_df.merge(region_df, on="region", how="left", validate="many_to_one")
print(f"  Loaded : {df.shape[0]:,} rows, {df.shape[1]} cols")

# ── Feature sets (must match train_model.ipynb STEP 2) ──────────────────────
LEAKY_COLS = ["legacy_propensity_score", "hist_enrollment_rate_region"]
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

CATEGORICAL_COLS = [
    "gender", "marital_status", "employment_type",
    "region", "has_dependents", "broker_channel",
]
usable_raw = set(df.columns) - ALL_FORBIDDEN
NUMERIC_COLS    = [c for c in usable_raw if c not in CATEGORICAL_COLS]
USABLE_FEATURES = NUMERIC_COLS + CATEGORICAL_COLS

print(f"\nStep 2 – Feature classification")
print(f"  Usable features : {len(USABLE_FEATURES)}")

# ── Encode ────────────────────────────────────────────────────────────────────
encoders = {}
df_enc = df.copy()
for col in CATEGORICAL_COLS:
    if col in df_enc.columns:
        df_enc[col] = df_enc[col].fillna("Missing").astype(str)
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col])
        encoders[col] = le

X = df_enc[USABLE_FEATURES].copy()
y = df["enrolled"].copy()

for col in NUMERIC_COLS:
    if col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nStep 3 – Train/Test split: {X_train.shape[0]:,} / {X_test.shape[0]:,}")

# ── Train ─────────────────────────────────────────────────────────────────────
print("\nStep 4 – Training XGBoost (no-leak model) …")
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

from sklearn.metrics import roc_auc_score
probs = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probs)
print(f"  AUC-ROC on test set: {auc:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
model_path   = os.path.join(MODEL_DIR, "xgb_noleak_tuned.joblib")
feature_path = os.path.join(MODEL_DIR, "feature_columns.json")

joblib.dump(model, model_path)
with open(feature_path, "w") as f:
    json.dump(X_train.columns.tolist(), f)

print(f"\nStep 5 – Artefacts saved")
print(f"  Model    -> {model_path}")
print(f"  Features -> {feature_path}")
print(f"  Done.")
