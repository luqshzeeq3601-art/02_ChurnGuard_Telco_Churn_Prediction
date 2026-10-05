# 08. Decisions Log (ADR-lite)

> One entry per decision. Never delete; supersede with a new entry.

| ID | Date | Decision | Reason | Status |
|---|---|---|---|---|
| D-001 | 05 Oct 2026 | Use IBM Telco Customer Churn dataset, framed as fictional Malaysian telco "NusaTel" | Public, clean, recognised by recruiters; churn matches AirAsia / Astro postings | Accepted |
| D-002 | 05 Oct 2026 | PR-AUC as primary metric | Imbalanced target; ranking churners matters most | Accepted |
| D-003 | 05 Oct 2026 | Blank `TotalCharges` -> 0 | All blanks have tenure = 0 (not yet billed) | Accepted |
| D-004 | 05 Oct 2026 | Keep "No internet / phone service" as separate category | Carries signal; avoids collapsing into "No" | Accepted |
| D-005 | 05 Oct 2026 | Default imbalance handling = scale_pos_weight / class weights; rejected SMOTE | SMOTE experiment (E06: 0.6537) degraded PR-AUC vs baseline (E05: 0.6598) and added training complexity | Accepted (Confirmed after E06) |
| D-006 | 05 Oct 2026 | Keep `gender` and `SeniorCitizen` in v1 feature set | E08 ablation confirmed minimal impact on PR-AUC (0.6732 -> 0.6706); retained for fairness slicing and auditing (NFR7) | Accepted (Confirmed after E08) |
| D-007 | 05 Oct 2026 | Select LightGBM (Optuna-tuned) as Champion model | Achieved top CV PR-AUC (0.6732), ROC-AUC (0.8468), Lift@10% (2.92x), and stable CV std (0.0217 <= 0.03) | Accepted (Confirmed after E07) |
| D-008 | 05 Oct 2026 | Deploy to GCP Cloud Run | Free tier; GCP appears in AirAsia and PetBacker postings | Accepted |
| D-009 | 05 Oct 2026 | Profit-based threshold, not 0.5 | Aligns model with retention budget decision | Accepted |
| D-010 | 05 Oct 2026 | Hybrid data: IBM Telco for modelling + data.gov.my for Malaysia context | No public customer-level Malaysian churn data (PDPA 2010) | Accepted |
| D-011 | 05 Oct 2026 | Publish on GitHub, Hugging Face Spaces (demo), Cloud Run (API), Kaggle, LinkedIn | Covers technical and non-technical recruiters; all free | Accepted |
| D-012 | 05 Oct 2026 | Streamlit dashboard promoted from Could to Must | Needed as public demo on Hugging Face Spaces | Accepted |
| D-013 | 05 Oct 2026 | Champion re-decision (LR vs LightGBM) using paired-fold rule in `14_IMPROVEMENT_PLAN.md` 4.4 | MO1 not met (+0.0145 PR-AUC vs +0.03 target, within CV noise) | Pending T9.6 |
| D-014 | 05 Oct 2026 | Fairness mitigation option (M0 / M1 / M2); no group-specific thresholds | Test recall gaps 0.078 (gender), 0.144 (SeniorCitizen) exceed NFR7 | Pending T9.7 |
| D-015 | 05 Oct 2026 | Test set reused exactly once for v1.1 after calibration and champion fixes; disclosed in README and model card | v1.0 calibration was in-sample and collapsed scores to 32 levels | Pending T9.9 |
| D-016 | 05 Oct 2026 | Reopen T0.1, T6.1 to T6.3 and T8.2 to T8.8; tasks ticked only with evidence | No git remote; Cloud Run, HF Space, Kaggle, LinkedIn, article not published | Accepted |
