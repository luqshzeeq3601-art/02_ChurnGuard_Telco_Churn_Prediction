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
| D-013 | 05 Oct 2026 | Champion re-decision: Logistic Regression selected as Champion, LightGBM as runner-up | Paired 5-fold CV (E11) on train+val: LightGBM mean paired PR-AUC gain over LR is +0.0045 (std 0.0078), which fails the pre-registered requirement (>= 1 std and >= 0.01). Simpler interpretable Logistic Regression selected; LightGBM retained as runner-up. MO1 (+0.03 target) is honestly NOT MET. | Accepted (Confirmed after E11) |
| D-014 | 05 Oct 2026 | Fairness mitigation: Option M2 (drop gender + SeniorCitizen) selected | E12 audit on OOF calibrated predictions at tau=0.1882 showed M2 achieves the lowest max recall gap (0.0744 vs M0: 0.0893, M1: 0.0909) with 0% profit loss (RM38,693 vs RM38,660). Gender gap is 0.0195 (meets NFR7 <= 0.05). Remaining senior gap (0.0744) is documented as known limitation driven by ground-truth base rate disparity (senior churn rate 41.3% vs non-senior 23.6%). Group-specific thresholds rejected. | Accepted (Confirmed after E12) |
| D-015 | 05 Oct 2026 | Test set reused exactly once for v1.1 after calibration and champion fixes; disclosed in README and model card | v1.0 calibration was in-sample and collapsed scores to 32 levels. Single test evaluation performed for v1.1 reporting Champion (LR M2) and Runner-up (LightGBM) side by side with 1,000x bootstrap CIs. | Accepted (Confirmed after T9.9) |
| D-016 | 05 Oct 2026 | Reopen T0.1, T6.1 to T6.3 and T8.2 to T8.8; tasks ticked only with evidence | No git remote; Cloud Run, HF Space, Kaggle, LinkedIn, article not published | Accepted |
| D-017 | 05 Oct 2026 | Track small public benchmark data (<1 MB) and serialized model in git | IBM Telco (CC0, 977 KB) and data.gov.my context (CC-BY 4.0, 1 KB) are small, public, and static. Tracking in git enables zero-dependency reproducible Docker builds, standalone CI runs without cloud bucket credentials, and seamless 1-click Hugging Face Space / Render deployment. Meets repo hygiene while maintaining self-containment. | Accepted |
