"""
agent_router.py
===============
Pure-Python query router for the enrollment outreach agent.

No LLM API is used.  The router uses keyword/pattern matching to map
natural-language queries to tool calls, formats the results, and enforces
all required agent behaviour:

  ✔ predict_enrollment    – model inference for one employee
  ✔ rank_outreach_candidates – prioritised outreach list per region
  ✔ lookup_region_profile – region-level stats
  ✔ explain_prediction    – natural-language explanation (fairness-guarded)
  ✔ validate_raw_row      – data quality check on a raw record

  ✔ Refuses to use legacy_propensity_score explicitly in responses
  ✔ Explanation never cites gender, marital_status, or age
  ✔ Supports at least 5 distinct demo query patterns
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from agent_tools import (
    predict_enrollment,
    rank_outreach_candidates,
    lookup_region_profile,
    explain_prediction,
    validate_raw_row,
    CONFIG,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  FORMATTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _hr(title: str = "", width: int = 70) -> str:
    if title:
        pad = (width - len(title) - 2) // 2
        return "=" * pad + f" {title} " + "=" * (width - pad - len(title) - 2)
    return "=" * width


def _fmt_dict(d: dict) -> str:
    lines = []
    for k, v in d.items():
        if isinstance(v, float):
            lines.append(f"  {k:<38}: {v:.4f}")
        else:
            lines.append(f"  {k:<38}: {v}")
    return "\n".join(lines)


def _fmt_df(df: pd.DataFrame, max_rows: int = 30) -> str:
    if "error" in df.columns:
        return f"  ERROR: {df['error'].iloc[0]}"
    display = df.head(max_rows)
    return display.to_string(index=False)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  INTENT PATTERNS
# ─────────────────────────────────────────────────────────────────────────────

# Regex patterns to detect query intent.
# Order matters: more specific patterns first.

_PATTERNS = [
    # ── Outreach ranking ──────────────────────────────────────────────────────
    (
        r"(top|best|priorit|rank|who.*contact|outreach)",
        "rank_outreach",
    ),
    # ── Explain why an employee is predicted to enroll ────────────────────────
    (
        r"(why|explain|reason|justif)",
        "explain",
    ),
    # ── Validate / check raw row ──────────────────────────────────────────────
    (
        r"(wrong|valid|check|problem|issue|bad.*row|raw.*row|what.*wrong)",
        "validate",
    ),
    # ── Region profile lookup ─────────────────────────────────────────────────
    (
        r"(region|profile|stat|capacity|benefit.*profile|premium|broker.*rating)",
        "region_profile",
    ),
    # ── Predict for a single employee ─────────────────────────────────────────
    (
        r"(predict|will.*enroll|enroll.*predict|score|probability)",
        "predict",
    ),
]

# Explicit leakage-related refusal triggers
_FORBIDDEN_MENTION_RE = re.compile(
    r"(legacy_propensity|propensity_score|hist_enrollment_rate)", re.IGNORECASE
)


def _detect_intent(query: str) -> str:
    q = query.lower()
    for pattern, intent in _PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return intent
    return "unknown"


def _extract_region(query: str) -> str | None:
    regions = ["midwest", "northeast", "south", "west"]
    q = query.lower()
    for r in regions:
        if r in q:
            return r.title()
    return None


def _extract_employee_id(query: str) -> int | None:
    # Look for patterns like "employee 12345", "emp 12345", "#12345", or bare integers
    m = re.search(
        r"(?:employee|emp|id|#)\s*[:=]?\s*(\d{4,6})", query, re.IGNORECASE
    )
    if m:
        return int(m.group(1))
    # Bare integer
    m2 = re.search(r"\b(\d{4,6})\b", query)
    if m2:
        return int(m2.group(1))
    return None


def _extract_top_k(query: str) -> int | None:
    m = re.search(r"top[\s-]?(\d+)", query, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MAIN ROUTER
# ─────────────────────────────────────────────────────────────────────────────

def ask(query: str, raw_row: dict[str, Any] | None = None) -> str:
    """
    Route a natural-language query to the appropriate agent tool.

    Parameters
    ----------
    query   : Natural-language question or command.
    raw_row : Optional raw employee dict, required when intent is 'validate'
              or when predicting from a raw record.

    Returns
    -------
    Formatted string response suitable for display in a notebook or terminal.
    """
    # ── Forbidden-feature refusal ─────────────────────────────────────────────
    if _FORBIDDEN_MENTION_RE.search(query):
        return (
            _hr("[REFUSAL]") + "\n"
            "This agent refuses to use or explain predictions that incorporate\n"
            "'legacy_propensity_score', 'hist_enrollment_rate_region', or any\n"
            "other field identified as directly reconstructing the enrollment target.\n\n"
            "These features were engineered from historical enrollment outcomes\n"
            "and would cause data leakage – producing misleadingly high accuracy\n"
            "during evaluation and unreliable predictions in production.\n\n"
            "Please restate your query without referencing those columns.\n"
            + _hr()
        )

    intent = _detect_intent(query)
    region = _extract_region(query)
    emp_id = _extract_employee_id(query)
    top_k  = _extract_top_k(query)

    # ─────────────────────────────────────────────────────────────────────────
    if intent == "rank_outreach":
        lines = [_hr(f"Outreach Ranking")]
        if region:
            df = rank_outreach_candidates(region=region, top_k=top_k)
            cap_msg = f"top {top_k}" if top_k else "capacity-limited"
            lines.append(
                f"  Region : {region}  |  Showing {cap_msg} candidates\n"
            )
        else:
            df = rank_outreach_candidates(region=None, top_k=top_k)
            lines.append("  All regions – respecting each region's hr_outreach_capacity\n")

        lines.append(_fmt_df(df, max_rows=50))
        lines.append(f"\n  Total candidates returned: {len(df)}")
        lines.append(_hr())
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    elif intent == "explain":
        if emp_id is None:
            return (
                "⚠ Please specify an employee ID.\n"
                "  Example: 'Why is employee 12345 predicted to enroll?'"
            )
        result = explain_prediction(emp_id)
        if "error" in result:
            return f"  ERROR: {result['error']}"

        lines = [_hr(f"Explanation – Employee {emp_id}")]
        lines.append(result["explanation"])
        lines.append("")
        lines.append(result["fairness_note"])
        if result.get("warnings"):
            lines.append("\n" + "\n".join(result["warnings"]))
        lines.append(_hr())
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    elif intent == "validate":
        if raw_row is None:
            return (
                "⚠ To validate a raw row, pass the record as the 'raw_row' "
                "argument to ask().\n"
                "  Example: agent.ask('what is wrong with this raw row?', raw_row=my_dict)"
            )
        result = validate_raw_row(raw_row)
        lines = [_hr("Raw Row Validation")]
        lines.append(f"  STATUS : {result['status']}")
        if result["issues"]:
            lines.append("\n  ISSUES FOUND:")
            for iss in result["issues"]:
                lines.append(f"    {iss}")
        else:
            lines.append("  No critical issues.")
        if result["warnings"]:
            lines.append("\n  WARNINGS:")
            for w in result["warnings"]:
                lines.append(f"    {w}")
        lines.append(_hr())
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    elif intent == "region_profile":
        if region is None:
            return (
                "⚠ Please specify a region name.\n"
                "  Known regions: Midwest, Northeast, South, West"
            )
        profile = lookup_region_profile(region)
        if "error" in profile:
            return f"  ERROR: {profile['error']}"

        lines = [_hr(f"Region Profile – {region}")]
        lines.append(_fmt_dict(profile))
        lines.append(_hr())
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    elif intent == "predict":
        if emp_id is None and raw_row is None:
            return (
                "⚠ Please specify an employee ID or pass a raw_row dict.\n"
                "  Example: 'Predict enrollment for employee 17825'"
            )
        result = predict_enrollment(
            employee_id=emp_id,
            raw_row=raw_row if emp_id is None else None,
        )
        if "error" in result:
            return f"  ERROR: {result['error']}"

        lines = [_hr(f"Prediction – Employee {result['employee_id']}")]
        lines.append(f"  Enrollment Probability : {result['enrollment_probability']:.4f}")
        lines.append(f"  Decision Threshold     : {result['threshold']}")
        lines.append(f"  Prediction             : {result['prediction']}")
        if result.get("warnings"):
            lines.append("\n" + "\n".join(result["warnings"]))
        lines.append(_hr())
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    else:
        return (
            "⚠ I could not understand that query.\n\n"
            "Supported query types:\n"
            "  • 'Who are the top 20 outreach priorities in the Midwest?'\n"
            "  • 'Rank outreach candidates across all regions'\n"
            "  • 'Why is employee 17825 predicted to enroll?'\n"
            "  • 'What is the region profile for the South?'\n"
            "  • 'Predict enrollment for employee 12324'\n"
            "  • 'What is wrong with this raw row?' (+ pass raw_row=...)\n"
            "  • 'Validate this record' (+ pass raw_row=...)\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CONVENIENCE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class EnrollmentAgent:
    """
    Thin wrapper around the ask() router so users can do:

        agent = EnrollmentAgent()
        print(agent.ask("who are the top 20 in Midwest?"))
    """

    def ask(self, query: str, raw_row: dict[str, Any] | None = None) -> str:
        response = ask(query, raw_row=raw_row)
        print(response)
        return response

    # ── Direct tool access for programmatic use ───────────────────────────────
    @staticmethod
    def predict(employee_id: int | None = None,
                raw_row: dict | None = None) -> dict:
        return predict_enrollment(employee_id=employee_id, raw_row=raw_row)

    @staticmethod
    def rank(region: str | None = None, top_k: int | None = None) -> pd.DataFrame:
        return rank_outreach_candidates(region=region, top_k=top_k)

    @staticmethod
    def region_profile(region: str) -> dict:
        return lookup_region_profile(region)

    @staticmethod
    def explain(employee_id: int) -> dict:
        return explain_prediction(employee_id)

    @staticmethod
    def validate(raw_row: dict) -> dict:
        return validate_raw_row(raw_row)
