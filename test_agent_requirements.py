# -*- coding: utf-8 -*-
"""
test_agent_requirements.py
==========================
Automated verification of all agent behaviour requirements.

Requirements tested
-------------------
 REQ-1  At least 5 distinct demo queries execute without error
 REQ-2  Agent refuses to use legacy_propensity_score (and similar leaky
         columns) when they appear in a query or in a raw_row payload
 REQ-3  explain_prediction never cites gender, marital_status, or age
         in the generated explanation text
 REQ-4  Architecture: pure-Python router (no LLM API), documented
 REQ-5  Integrity: every refusal is explicit in the response

Run:
    python test_agent_requirements.py          # coloured pass/fail summary
    pytest test_agent_requirements.py -v       # via pytest
"""
from __future__ import annotations

import re
import sys
import os
import traceback
from typing import Callable

# Force UTF-8 output on Windows so Unicode chars don't crash cp1252
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
# pyrefly: ignore [missing-import]
from agent_router import EnrollmentAgent, ask  # noqa: E402

# ── helpers ───────────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

PASS_MARK = "[PASS]"
FAIL_MARK = "[FAIL]"

_results: list[tuple[str, bool, str]] = []


def _banner(title: str) -> None:
    print(f"\n{CYAN}{'='*70}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{CYAN}{'='*70}{RESET}")


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = f"{GREEN}{PASS_MARK}{RESET}" if condition else f"{RED}{FAIL_MARK}{RESET}"
    _results.append((name, condition, detail))
    print(f"  {status}  {name}")
    if detail and not condition:
        for line in detail.splitlines():
            print(f"       {YELLOW}{line}{RESET}")
    elif detail:
        for line in detail.splitlines()[:3]:          # show first 3 lines on pass
            print(f"       {line}")
    return condition


def run_test(fn: Callable) -> None:
    """Run a test function, catching any exception as a FAIL."""
    try:
        fn()
    except Exception as exc:
        tb = traceback.format_exc(limit=4)
        check(f"[EXCEPTION in {fn.__name__}]", False, tb)


# ── load a real employee ID from the processed CSV ────────────────────────────
_emp_csv = os.path.join(
    os.path.dirname(__file__), "data", "processed",
    "employees_processed_no_leaky_features.csv",
)
_sample_id = int(pd.read_csv(_emp_csv, usecols=["employee_id"]).iloc[1]["employee_id"])
_sample_id2 = int(pd.read_csv(_emp_csv, usecols=["employee_id"]).iloc[50]["employee_id"])

agent = EnrollmentAgent()

# ─────────────────────────────────────────────────────────────────────────────
# REQ-1: At least 5 distinct demo queries
# ─────────────────────────────────────────────────────────────────────────────

def test_query1_outreach_midwest():
    _banner("REQ-1 | Query 1 – Top 20 outreach priorities in the Midwest")
    resp = agent.ask("Who are the top 20 outreach priorities in the Midwest this window?")
    has_content = len(resp) > 50
    has_rank = "outreach_rank" in resp.lower() or "midwest" in resp.lower() or "enrollment" in resp.lower()
    check("Response is non-empty", has_content, resp[:200])
    check("Response references Midwest / enrollment probabilities", has_rank, resp[:200])


def test_query2_explain_why():
    _banner(f"REQ-1 | Query 2 – Why is employee {_sample_id} predicted to enroll?")
    resp = agent.ask(f"Why is employee {_sample_id} predicted to enroll?")
    has_content = len(resp) > 50
    mentions_employee = str(_sample_id) in resp
    check("Response is non-empty", has_content, resp[:200])
    check("Response mentions the employee ID", mentions_employee, resp[:200])


def test_query3_validate_bad_row():
    _banner("REQ-1 | Query 3 – What is wrong with this raw row?")
    bad_row = {
        "employee_id":             99999,
        "legacy_propensity_score": 0.92,
        "salary_clean":            -5000,
        "employment_type":         "Full-time",
        "region":                  "Midwest",
        "has_dependents":          "Yes",
        "broker_channel":          "Direct",
        "tenure_years":            2.5,
    }
    resp = agent.ask("What is wrong with this raw row?", raw_row=bad_row)
    has_content = len(resp) > 30
    mentions_issue = any(kw in resp.upper() for kw in ["ISSUE", "FAIL", "LEAKAGE", "INVALID", "MISSING"])
    check("Validation response is non-empty", has_content, resp[:300])
    check("Validation reports at least one issue", mentions_issue, resp[:300])


def test_query4_region_profile():
    _banner("REQ-1 | Query 4 – Region profile for the South")
    resp = agent.ask("What is the region profile for the South?")
    has_content = len(resp) > 50
    has_region_data = "south" in resp.lower() or "premium" in resp.lower() or "capacity" in resp.lower()
    check("Region profile response is non-empty", has_content, resp[:200])
    check("Response contains region statistics", has_region_data, resp[:200])


def test_query5_predict_enrollment():
    _banner(f"REQ-1 | Query 5 – Predict enrollment for employee {_sample_id}")
    resp = agent.ask(f"Predict enrollment probability for employee {_sample_id}")
    has_content = len(resp) > 30
    has_probability = "probability" in resp.lower() or "enroll" in resp.lower()
    check("Prediction response is non-empty", has_content, resp[:200])
    check("Response contains enrollment probability", has_probability, resp[:200])


def test_query6_all_regions_ranking():
    _banner("REQ-1 | Query 6 (Bonus) – Rank outreach candidates across all regions")
    resp = agent.ask("Rank outreach candidates across all regions using their capacity")
    has_content = len(resp) > 50
    check("All-region ranking response is non-empty", has_content, resp[:200])


# ─────────────────────────────────────────────────────────────────────────────
# REQ-2: Explicit refusal for legacy_propensity_score (and similar leaky cols)
# ─────────────────────────────────────────────────────────────────────────────

def test_refusal_legacy_in_query():
    _banner("REQ-2 | Refusal – legacy_propensity_score mentioned in query text")
    resp = agent.ask(
        "Can you predict using the legacy_propensity_score for employee 12345?"
    )
    # Must explicitly refuse
    is_refusal = "refus" in resp.lower()
    mentions_leakage = "leakage" in resp.lower() or "forbidden" in resp.lower() or "leak" in resp.lower()
    check("Response contains explicit REFUSAL", is_refusal, resp[:400])
    check("Refusal explains data leakage concern", mentions_leakage, resp[:400])


def test_refusal_legacy_in_raw_row():
    _banner("REQ-2 | Refusal – legacy_propensity_score silently passed in raw_row")
    row_with_leakage = {
        "employee_id":               55555,
        "legacy_propensity_score":   0.87,  # forbidden
        "hist_enrollment_rate_region": 0.62, # also forbidden
        "salary_clean":              72_000,
        "employment_type":           "Full-time",
        "region":                    "West",
        "has_dependents":            "No",
        "broker_channel":            "Employer-Sponsored",
        "tenure_years":              3.1,
        "salary_band":               3,
        "salary_per_tenure":         23_225.8,
        "salary_age_ratio":          2_000.0,
        "is_new_hire":               0,
        "prior_year_enrolled_clean": 0,
        "has_outreach_note":         1,
    }
    resp = agent.ask("Predict enrollment for this employee", raw_row=row_with_leakage)
    mentions_refusal_or_warning = any(
        kw in resp.upper() for kw in ["REFUSAL", "FORBIDDEN", "LEAKAGE", "EXCLUDED", "REMOVED"]
    )
    check(
        "Response flags the forbidden column when passed in raw_row",
        mentions_refusal_or_warning,
        resp[:400],
    )


def test_refusal_hist_enrollment_in_query():
    _banner("REQ-2 | Refusal – hist_enrollment_rate mentioned in query text")
    resp = agent.ask(
        "Use hist_enrollment_rate to boost predictions this window"
    )
    is_refusal = "refus" in resp.lower()
    check("Response explicitly refuses hist_enrollment_rate query", is_refusal, resp[:300])


def test_prediction_does_not_use_forbidden_cols():
    _banner("REQ-2 | Integrity – Model feature list must not contain forbidden columns")
    import json
    feature_list_path = os.path.join(
        os.path.dirname(__file__), "models", "feature_columns.json"
    )
    with open(feature_list_path) as f:
        features = json.load(f)

    FORBIDDEN = {
        "legacy_propensity_score",
        "hist_enrollment_rate_region",
        "contact_to_application_days",
        "days_contact_to_app",
        "is_contact_after_app",
        "has_applied",
        "app_month",
        "app_day_of_week",
        "application_date",
        "plan_tier_requested",
    }
    leaky_in_model = [f for f in features if f in FORBIDDEN]
    check(
        "No forbidden / leaky features in saved feature_columns.json",
        len(leaky_in_model) == 0,
        f"Found: {leaky_in_model}" if leaky_in_model else "Clean ✔",
    )


# ─────────────────────────────────────────────────────────────────────────────
# REQ-3: explain_prediction must not cite gender, marital_status, or age
# ─────────────────────────────────────────────────────────────────────────────

_DEMO_PROTECTED_TERMS = re.compile(
    r"\b(gender|marital.?status|married|single|divorced|male|female|non.?binary"
    r"|age(?!\s+ratio)|years\s+old)\b",
    re.IGNORECASE,
)


def _explanation_cites_protected(explanation: str) -> list[str]:
    """Return list of protected terms found in the explanation text."""
    return _DEMO_PROTECTED_TERMS.findall(explanation)


def test_explain_no_protected_attrs_primary():
    _banner(f"REQ-3 | Fairness – Explanation for employee {_sample_id} omits protected attrs")
    # pyrefly: ignore [missing-import]
    from agent_tools import explain_prediction
    result = explain_prediction(_sample_id)

    if "error" in result:
        check("explain_prediction returned successfully", False, result["error"])
        return

    explanation = result.get("explanation", "")
    hits = _explanation_cites_protected(explanation)
    check(
        "Explanation text does NOT cite gender / marital_status / age",
        len(hits) == 0,
        f"Protected terms found: {hits}\n\nExplanation:\n{explanation}",
    )
    # Confirm fairness note is present
    fairness_note = result.get("fairness_note", "")
    has_fairness_note = len(fairness_note) > 10
    check("Fairness disclaimer is included in the response", has_fairness_note, fairness_note)


def test_explain_no_protected_attrs_secondary():
    _banner(f"REQ-3 | Fairness – Explanation for employee {_sample_id2} omits protected attrs")
    # pyrefly: ignore [missing-import]
    from agent_tools import explain_prediction
    result = explain_prediction(_sample_id2)

    if "error" in result:
        check("explain_prediction returned successfully", False, result["error"])
        return

    explanation = result.get("explanation", "")
    hits = _explanation_cites_protected(explanation)
    check(
        f"Explanation for employee {_sample_id2} does NOT cite protected attrs",
        len(hits) == 0,
        f"Protected terms found: {hits}\n\nExplanation:\n{explanation}",
    )


def test_explain_via_router_no_protected_attrs():
    """
    The router response includes two parts separated by the fairness note:
      1. The explanation body   – must NOT cite protected attrs
      2. The fairness disclaimer – intentionally names 'gender, marital status, age'
                                   to explain what it omitted (this is correct behaviour)

    We test only part 1 so we don't false-positive on the disclaimer itself.
    """
    _banner("REQ-3 | Fairness – Router-level explanation body omits protected attrs")
    resp = agent.ask(f"Why is employee {_sample_id} predicted to enroll?")

    # Split on the NOTE: line – everything before it is the explanation body
    note_marker = "NOTE:"
    if note_marker in resp:
        body = resp[:resp.index(note_marker)]
        disclaimer = resp[resp.index(note_marker):]
    else:
        body = resp
        disclaimer = ""

    hits_in_body = _explanation_cites_protected(body)
    check(
        "Explanation BODY does NOT cite protected attrs (gender/marital/age)",
        len(hits_in_body) == 0,
        f"Protected terms found in body: {hits_in_body}\n\nBody:\n{body}",
    )
    # Confirm the fairness disclaimer is present (it is OK for it to name the attributes)
    has_disclaimer = len(disclaimer) > 10
    check(
        "Fairness disclaimer is present in the router response",
        has_disclaimer,
        disclaimer[:200],
    )


# ─────────────────────────────────────────────────────────────────────────────
# REQ-4: Architecture – pure-Python router, no LLM API
# ─────────────────────────────────────────────────────────────────────────────

def test_no_llm_api_in_source():
    _banner("REQ-4 | Architecture – No LLM API calls in source files")
    LLM_PATTERNS = [
        "openai", "anthropic", "cohere", "requests.post", "langchain",
        "ChatCompletion", "groq", "gemini", "vertexai",
    ]
    src_dir = os.path.join(os.path.dirname(__file__), "src")
    violations: dict[str, list[str]] = {}

    for fname in os.listdir(src_dir):
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(src_dir, fname)
        with open(fpath, encoding="utf-8") as f:
            content = f.read().lower()
        found = [p for p in LLM_PATTERNS if p.lower() in content]
        if found:
            violations[fname] = found

    if violations:
        detail = "\n".join(f"  {k}: {v}" for k, v in violations.items())
    else:
        detail = "No LLM API imports found – pure Python router confirmed ✔"

    check("No LLM API imports in src/ Python files", len(violations) == 0, detail)


def test_five_tool_functions_exist():
    _banner("REQ-4 | Architecture – All 5 required tool functions are importable")
    # pyrefly: ignore [missing-import]
    from agent_tools import (
        predict_enrollment,
        rank_outreach_candidates,
        lookup_region_profile,
        explain_prediction,
        validate_raw_row,
    )
    tools = [
        predict_enrollment, rank_outreach_candidates,
        lookup_region_profile, explain_prediction, validate_raw_row,
    ]
    check(
        "All 5 tool functions exist and are callable",
        all(callable(t) for t in tools),
        "predict_enrollment, rank_outreach_candidates, lookup_region_profile, "
        "explain_prediction, validate_raw_row",
    )


# ─────────────────────────────────────────────────────────────────────────────
# REQ-5: Refusal is EXPLICIT in the response text
# ─────────────────────────────────────────────────────────────────────────────

def test_refusal_is_explicit_not_silent():
    _banner("REQ-5 | Integrity – Refusal response contains explicit human-readable message")
    resp = agent.ask("Give me a score using legacy_propensity_score")
    # The word "refus" must appear AND the user must understand WHY
    has_refusal_word = "refus" in resp.lower()
    has_reason = any(
        kw in resp.lower()
        for kw in ["leakage", "reconstruct", "target", "forbidden", "mislead"]
    )
    check("Refusal uses the word 'refuse' / 'refusal'", has_refusal_word, resp[:400])
    check("Refusal explains WHY (leakage / forbidden reason)", has_reason, resp[:400])


def test_validate_flags_forbidden_explicitly():
    _banner("REQ-5 | Integrity – validate_raw_row explicitly names forbidden columns")
    # pyrefly: ignore [missing-import]
    from agent_tools import validate_raw_row
    row = {
        "employee_id": 12345,
        "legacy_propensity_score": 0.7,
        "hist_enrollment_rate_region": 0.45,
        "salary_clean": 55000,
    }
    result = validate_raw_row(row)
    issues_text = " ".join(result.get("issues", []))
    has_explicit_name = (
        "legacy_propensity_score" in issues_text.lower()
        or "forbidden" in issues_text.lower()
        or "leakage" in issues_text.lower()
    )
    check(
        "validate_raw_row explicitly names the forbidden column in issues list",
        has_explicit_name,
        f"Issues: {result.get('issues', [])}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: run all tests and print summary
# ─────────────────────────────────────────────────────────────────────────────

_ALL_TESTS = [
    # REQ-1
    test_query1_outreach_midwest,
    test_query2_explain_why,
    test_query3_validate_bad_row,
    test_query4_region_profile,
    test_query5_predict_enrollment,
    test_query6_all_regions_ranking,
    # REQ-2
    test_refusal_legacy_in_query,
    test_refusal_legacy_in_raw_row,
    test_refusal_hist_enrollment_in_query,
    test_prediction_does_not_use_forbidden_cols,
    # REQ-3
    test_explain_no_protected_attrs_primary,
    test_explain_no_protected_attrs_secondary,
    test_explain_via_router_no_protected_attrs,
    # REQ-4
    test_no_llm_api_in_source,
    test_five_tool_functions_exist,
    # REQ-5
    test_refusal_is_explicit_not_silent,
    test_validate_flags_forbidden_explicitly,
]


def main():
    print(f"\n{BOLD}{'='*70}")
    print("  Froncort AI – Outreach Agent  |  Requirement Verification Suite")
    print(f"{'='*70}{RESET}")

    for fn in _ALL_TESTS:
        run_test(fn)

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)
    total  = len(_results)

    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  SUMMARY  |  {GREEN}{passed} passed{RESET}{BOLD}  ·  "
          f"{RED if failed else RESET}{failed} failed{RESET}{BOLD}  /  {total} checks{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")

    if failed:
        print(f"\n{RED}Failed checks:{RESET}")
        for name, ok, detail in _results:
            if not ok:
                print(f"  {RED}{FAIL_MARK}{RESET} {name}")
        sys.exit(1)
    else:
        print(f"\n{GREEN}All checks passed!{RESET}\n")


# ── pytest compatibility (each _test_* function is auto-collected) ────────────
# When run via pytest, tests are discovered automatically.
# When run directly, main() fires.

if __name__ == "__main__":
    main()
