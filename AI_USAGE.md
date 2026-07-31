# AI Usage Disclosure Statement (`AI_USAGE.md`)

This document discloses the use of Artificial Intelligence (AI) tools and assistants during the end-to-end architecture, development, data auditing, model engineering, and agent system integration for the **Insurance Enrollment Prediction & Outreach Assistant Agent**.

---

## Executive Summary & Tooling Overview

AI tools were leveraged throughout the project lifecycle as collaborative engineering and analytical assistants to implement the logic for the project.

### AI Tooling Stack

The primary tools utilized across phases include:

* **GitHub Copilot:**
* **ChatGPT (OpenAI):**
* **Claude 3.5 Sonnet (Anthropic):**
* **Gemini Pro (Google):**
* **AntiGravity IDE Assistance:**

---

## Detailed Stage-by-Stage AI Assistance

### Stage 1. Data Preprocessing & Feature Engineering Assistance

#### AI Tools Employed
* **ChatGPT**, **Gemini Pro**, **AntiGravity IDE Assistance**

AI assistance during the data preparation phase was focused primarily on syntax optimization, pandas transformation efficiency, and assistance in feature engineering logic. The tools helped in standardizing dirty categorical values, parsing mixed date formats, and reinterpreting tricky sentinel flags like `prior_year_enrolled`. Additionally, AI supported the sentiment extraction from unstructured outreach notes, and out-of-fold regional aggregate target encoding ensuring clean, efficient pipeline transformations while eliminating in-place pandas bugs.


---

### Stage 2. Model Development, Debugging & Hyperparameter Tuning

#### AI Tools Employed
* **ChatGPT**, **Gemini Pro**

In the model engineering phase, AI tools served as diagnostic and optimization assistants to streamline XGBoost classifier training. Assistance was used to debug pipeline errors, cross checking gradient boosting parameters, and structure cross-validation workflows. The AI also provided guidance on checking the potential for data leakage across feature groups.


---

### Stage 3: AI Agent for Outreach Purposes

#### AI Tools Employed
* **Claude 3.5 Sonnet**, **Gemini Pro**

For the agentic outreach layer, AI tools scaffolded the core system architecture, and state-routing mechanics based on functional specifications.
Human involvement was maintained at the supervisory level by supplying the initial project requirements, defining tool capabilities, and providing monitoring. The AI implemented deterministic tool interfaces (`predict_enrollment`, `rank_outreach_candidates`, `lookup_region_profile`, and `explain_prediction`) alongside hard refusal guardrails to reject leaky predictors and censor protected attributes from natural-language explanations.


---
### AI Agent for Documentation purposes

#### AI Tools Employed
* **Claude 3.5 Sonnet**, **Gemini Pro**

For the documentation and reporting purposes, AI tools were used to modify the technical report and present the project findings.
