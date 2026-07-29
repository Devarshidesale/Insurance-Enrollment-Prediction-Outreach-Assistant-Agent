# AI Usage Log: Insurance Enrollment Prediction & Outreach Assistant Agent

## Purpose of This Document

This log documents how AI/LLM tooling was used to support the architecture, documentation, and validation design for the project. It is intended to provide transparency, reproducibility, and governance traceability.

---

## 1) AI-Assisted Activities Performed

### 1.1 System Architecture Design Support
AI assistance was used to:
- Structure the end-to-end architecture from raw data ingestion through outreach agent decisions
- Organize module boundaries (`data_prep`, `feature_engineering`, `train_model`, `agent_tools`, `agent_router`)
- Draft flow-level representation linking model scoring to regional outreach capacity constraints

### 1.2 Data Audit and Cleaning Strategy Formulation
AI support was used to synthesize robust cleaning decisions for:
- Duplicate `employee_id` conflict resolution
- Date normalization across heterogeneous string formats
- Categorical normalization for outreach channels
- Sentinel reinterpretation logic for `prior_year_enrolled = 1`
- Join integrity controls for `region` key linkage

### 1.3 Compliance and Feature Governance Design
AI assistance was used to formalize:
- Explicit prohibition of `legacy_propensity_score`
- Exclusion of regional target aggregates (`hist_enrollment_rate_region`)
- Demographic exclusion checkpoint (`gender`, `marital_status`, `age`)
- Refusal and filtering policies for agent outputs

### 1.4 Evaluation and Business Metric Framing
AI support was used to define metric governance, including:
- Baseline model comparisons (majority class and simple heuristic)
- Core statistical metrics (AUC-ROC, Log-Loss)
- Operational ranking metrics (Precision@K, Lift@K) bounded by `hr_outreach_capacity`

### 1.5 Documentation Authoring and Review
AI assistance was used to:
- Draft complete project README and technical report content
- Improve consistency of terminology across files
- Ensure policy language is explicit and auditable

---

## 2) Validation Procedures Applied to AI Outputs

All AI-generated content was reviewed for:
1. **Domain consistency:** alignment with insurance enrollment outreach use case
2. **Policy correctness:** explicit treatment of leakage and demographic guardrails
3. **Operational realism:** capacity-aware ranking emphasis
4. **Traceability:** clear sectioning and rationale for major decisions

Additional checks performed:
- Verification that no implementation Python code was generated
- Verification that module files are stubs only
- Verification that required configuration files include requested dependency and ignore entries

---

## 3) Human Oversight and Decision Accountability

AI outputs were treated as drafting support, not autonomous final decisions. Final responsibility remained with engineering oversight for:
- Accepting or rejecting architectural recommendations
- Ensuring compliance framing is explicit and enforceable
- Confirming documentation completeness and enterprise appropriateness

Any future production implementation should maintain this human-in-the-loop standard for model governance, legal review, and policy approval.

---

## 4) Known Risks of AI Assistance and Mitigations

### Risk: Overgeneralized guidance
- **Mitigation:** Constrain outputs to project-specific fields and policies.

### Risk: Implicit assumptions not suitable for production
- **Mitigation:** Require explicit rationale and audit-ready wording in technical documentation.

### Risk: Compliance ambiguity
- **Mitigation:** State hard refusal rules and prohibited-feature policies in unambiguous terms.

### Risk: False confidence from polished text
- **Mitigation:** Validate against measurable checkpoints and downstream implementation requirements.

---

## 5) Allowed and Disallowed AI Contribution Scope

### Allowed in this deliverable
- Documentation drafting
- Architecture structuring
- Policy and validation framework articulation
- Quality and consistency refinement

### Explicitly disallowed in this deliverable
- Generation of Python implementation code
- Generation of executable model training or inference scripts
- Injection of synthetic experimental results presented as factual outcomes

---

## 6) Reproducibility Notes

To reproduce this documentation workflow:
1. Start from the defined project problem statement and governance constraints.
2. Draft architecture and compliance checkpoints before implementation details.
3. Validate that all required markdown/config files are complete and non-placeholder.
4. Confirm module files remain stubs until implementation phase begins.

This process ensures separation between design governance artifacts and future coding execution.
