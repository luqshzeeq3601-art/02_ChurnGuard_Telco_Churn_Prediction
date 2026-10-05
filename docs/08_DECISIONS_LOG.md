# 08. Decisions Log (ADR-lite)

> One entry per decision. Never delete; supersede with a new entry.

| ID | Date | Decision | Reason | Status |
|---|---|---|---|---|
| D-001 | 05 Oct 2026 | Use IBM Telco Customer Churn dataset, framed as fictional Malaysian telco "NusaTel" | Public, clean, recognised by recruiters; churn matches AirAsia / Astro postings | Accepted |
| D-002 | 05 Oct 2026 | PR-AUC as primary metric | Imbalanced target; ranking churners matters most | Accepted |
| D-003 | 05 Oct 2026 | Blank `TotalCharges` -> 0 | All blanks have tenure = 0 (not yet billed) | Accepted |
| D-004 | 05 Oct 2026 | Keep "No internet / phone service" as separate category | Carries signal; avoids collapsing into "No" | Accepted |
| D-005 | 05 Oct 2026 | Default imbalance handling = class weights + threshold tuning; SMOTE only as experiment E06 | Simpler, no synthetic data in serving path | Accepted (revisit after E06) |
| D-006 | 05 Oct 2026 | Keep `gender` and `SeniorCitizen` in v1, run ablation E08 | Decide based on performance and fairness evidence | Pending E08 |
| D-007 | 05 Oct 2026 | LightGBM as expected champion, final choice by CV | Strong on tabular data, fast, common in postings | Pending E05/E07 |
| D-008 | 05 Oct 2026 | Deploy to GCP Cloud Run | Free tier; GCP appears in AirAsia and PetBacker postings | Accepted |
| D-009 | 05 Oct 2026 | Profit-based threshold, not 0.5 | Aligns model with retention budget decision | Accepted |
| D-010 | 05 Oct 2026 | Hybrid data: IBM Telco for modelling + data.gov.my for Malaysia context | No public customer-level Malaysian churn data (PDPA 2010) | Accepted |
| D-011 | 05 Oct 2026 | Publish on GitHub, Hugging Face Spaces (demo), Cloud Run (API), Kaggle, LinkedIn | Covers technical and non-technical recruiters; all free | Accepted |
| D-012 | 05 Oct 2026 | Streamlit dashboard promoted from Could to Must | Needed as public demo on Hugging Face Spaces | Accepted |
