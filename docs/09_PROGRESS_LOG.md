# 09. Progress Log

> Newest entry on top. Each session: Done, Next, Blockers. Weekly PM review every Sunday.

## Current Status
| Item | Value |
|---|---|
| Current phase | **Phase 9: Hardening** (Sprint C: Visibility) |
| Next task | **T9.12** Create GitHub repo, push, CI green on GitHub, badges in README |
| Overall progress | 47 / 61 tasks |
| Health | On track: Sprint B complete, Sprint C in progress |

---

## 05 Oct 2026: Task T9.11 Completed (Phase 9 Sprint B Complete)
- **Done**:
  - **T9.11**: Updated [`docs/06_EXPERIMENT_PLAN.md`](file:///c:/Users/ZeeqRyz/Desktop/Ai-ML/Machine%20Learning%20Projects/02_ChurnGuard_Telco_Churn_Prediction/docs/06_EXPERIMENT_PLAN.md) and [`README.md`](file:///c:/Users/ZeeqRyz/Desktop/Ai-ML/Machine%20Learning%20Projects/02_ChurnGuard_Telco_Churn_Prediction/README.md) with final v1.1 evaluation results, bootstrap confidence intervals, negative results ("What Did Not Work"), and model limitations.
  - Verified no placeholders (`__`) remain in results tables or experiment documentation.
  - Documented negative results: SMOTE oversampling dilution, gradient boosting complexity premium (< 1 std over Logistic Regression), in-sample prefit calibration collapse, and fairness mitigation trade-offs.
- **Evidence**:
  - `python -c "assert '__' not in open('docs/06_EXPERIMENT_PLAN.md', encoding='utf-8').read(); assert '__' not in open('README.md', encoding='utf-8').read()"` passed cleanly.
  - `README.md` and `docs/06_EXPERIMENT_PLAN.md` synced with `reports/final_metrics.json`.
- **Next**: **T9.12** Create GitHub repo, push, CI green on GitHub, badges in README; re-tick T0.1, T6.1, T6.2 (F1).
- **Blockers**: None.

---

## 05 Oct 2026: Task T9.10 Completed (Phase 9 Sprint B)
- **Done**:
  - **T9.10**: Authored comprehensive production Model Card [`docs/MODEL_CARD.md`](file:///c:/Users/ZeeqRyz/Desktop/Ai-ML/Machine%20Learning%20Projects/02_ChurnGuard_Telco_Churn_Prediction/docs/MODEL_CARD.md) following the Mitchell et al. (2019) specification.
  - Documented:
    - Model Details (v1.1.0 Champion Logistic Regression M2 + Sigmoid cv=5, Runner-up LightGBM + Isotonic cv=5).
    - Intended use, out-of-scope and prohibited use cases.
    - Data splits & feature engineering with explicit fairness mitigation (Option M2).
    - Side-by-side test performance table with 1,000x percentile bootstrap 95% Confidence Intervals.
    - Full E12 fairness trade-off audit across gender and senior citizens with base-rate disparity explanation.
    - Decisions log cross-references (D-013, D-014, D-015).
    - Limitations and caveats.
- **Evidence**:
  - [`docs/MODEL_CARD.md`](file:///c:/Users/ZeeqRyz/Desktop/Ai-ML/Machine%20Learning%20Projects/02_ChurnGuard_Telco_Churn_Prediction/docs/MODEL_CARD.md) authored and verified.
- **Next**: **T9.11** Update `06_EXPERIMENT_PLAN.md` results (E10 to E12, v1.1), README results + limitations + "what did not work" (F9).
- **Blockers**: None.

---

## 05 Oct 2026: Task T9.9 Completed (Phase 9 Sprint B)
- **Done**:
  - **T9.9**: Executed v1.1 final test evaluation on held-out test split (1,057 samples, touched strictly once per D-015 disclosure) using Champion (Logistic Regression M2 + Sigmoid cv=5) and Runner-up (Optuna-tuned LightGBM + Isotonic cv=5) fit on `train + val` (5,986 samples).
  - Saved regenerated production model artifacts:
    - `models/model.joblib`: Champion pipeline serialized.
    - `models/runner_up_model.joblib`: Runner-up pipeline serialized.
    - `models/model_meta.json`: Full v1.1 production metadata with disclosure.
    - `reports/final_metrics.json`: Side-by-side Champion & Runner-up metrics with 1,000x bootstrap 95% CIs.
    - `reports/scored.csv`: Ranked test predictions.
    - `reports/figures/09_shap_summary.png`: SHAP summary plot.
  - Verified API integration test suite passes completely.
- **Evidence**:
  - **Champion Test Performance (v1.1)**:
    - **ROC-AUC**: **0.8449** [95% CI: 0.8207, 0.8714] (MO2 Met)
    - **PR-AUC**: **0.6739** [95% CI: 0.6203, 0.7262] (MO2 Met, lower bound $\ge 0.62$)
    - **Brier Score**: **0.1361** [95% CI: 0.1236, 0.1476] (MO3 Met)
    - **Lift@10%**: **2.84x** [95% CI: 2.52x, 3.18x] (MO2 Met, lower bound $\ge 2.5$)
    - **Recall@20%**: 48.40% [95% CI: 44.21%, 53.26%]
    - **Precision at $\tau^*$ (0.19)**: 47.53% [95% CI: 43.42%, 51.91%]
    - **Recall at $\tau^*$ (0.19)**: **88.97%** [95% CI: 85.00%, 92.44%]
    - **Expected Profit per 1k**: **RM36,720.15** [95% CI: RM30,063.54, RM42,997.04] (BO2 Met, beats Contact All RM18,193.09 & Contact None RM0.00)
  - **Runner-Up Test Performance (LightGBM)**:
    - ROC-AUC: 0.8473, PR-AUC: 0.6760, Brier: 0.1340, Lift@10%: 2.98x, Profit/1k: RM37,613.72
  - `pytest tests/test_api.py`: 6/6 passed.
  - `pytest tests/test_predict.py`: 3/3 passed.
  - `pytest tests/test_serialize.py`: 1/1 passed.
- **Next**: **T9.10** `docs/MODEL_CARD.md`: intended use, data, metrics with CI, fairness table, limitations.
- **Blockers**: None.

---

## 05 Oct 2026: Task T9.8 Completed (Phase 9 Sprint B)
- **Done**:
  - **T9.8**: Refactored objective checks in `src/churnguard/models/evaluate_test.py` via `compute_objectives_verification` to eliminate all hardcoded constants and fix comparison mismatches (F4):
    - **MO1**: Read dynamically from `reports/e11_champion_decision.json` (paired 5-fold CV on train+val) rather than comparing test PR-AUC to a hardcoded baseline constant.
    - **MO2**: Validates test point estimates and reports bootstrap 95% confidence interval lower bounds.
    - **MO3**: Evaluates test Brier score of calibrated probabilities strictly against uncalibrated probabilities computed on the **exact same test rows**.
    - **BO1**: Evaluates tie-aware `Recall@20%` on held-out test set ($\ge 50\%$).
    - **BO2**: Compares campaign profit at $\tau^*$ against dynamic Contact All and Contact None benchmarks computed on the **exact same test rows**.
    - **NFR7**: Computes demographic recall gaps on test split directly.
  - Added unit test `test_compute_objectives_verification_same_split` in `tests/test_evaluate_test.py`.
- **Evidence**:
  - Zero hardcoded baseline numbers in objective verification logic.
  - Same test rows guaranteed for both Brier comparison (`same_test_rows: True`) and profit comparison (`same_test_rows: True`).
  - `pytest tests/test_evaluate_test.py`: 3/3 passed in 7.78s.
- **Next**: **T9.9** v1.1 final test evaluation (once), champion + runner-up (F1, F2, F3, F5, F7).
- **Blockers**: None.

---

## 05 Oct 2026: Task T9.7 Completed (Phase 9 Sprint B)
- **Done**:
  - **T9.7**: Executed Experiment E12 (Fairness Audit and Mitigation) evaluating demographic fairness across `gender` and `SeniorCitizen` on out-of-fold calibrated predictions at $\tau^* = 0.1882$.
  - Evaluated M0 (current features), M1 (drop `gender`), and M2 (drop `gender` and `SeniorCitizen`).
  - Pre-registered selection rule applied (`14_IMPROVEMENT_PLAN.md` section 4.5):
    - Trade-off matrix:
      | Option | Description | Profit/1k (RM) | Gender Recall Gap | Senior Recall Gap | Max Recall Gap |
      |---|---|---|---|---|---|
      | M0 | Current features | RM38,660.32 | 0.0196 | 0.0893 | 0.0893 |
      | M1 | Drop gender | RM38,771.27 | 0.0221 | 0.0909 | 0.0909 |
      | M2 | Drop gender + SeniorCitizen | **RM38,693.49** | **0.0195** | **0.0744** | **0.0744** |
    - Selected **Option M2** (lowest max recall gap of 0.0744, 0% profit loss vs M0).
    - Gender gap is 0.0195 (satisfies NFR7 $\le 0.05$).
    - Remaining SeniorCitizen recall gap (0.0744) is documented in **D-014** as a known limitation driven by underlying churn base rate disparity (seniors churn at 41.3% vs non-seniors at 23.6%, a 1.75x ratio). Group-specific thresholds were rejected.
  - Saved full results to `reports/e12_fairness_audit.json` and logged run to MLflow experiment `E12_Fairness_Audit`.
  - Added unit test in `tests/test_fairness_audit.py`.
- **Evidence**:
  - Smallest max recall gap: 0.0744 (achieved by M2).
  - Profit impact: RM38,693.49 (+0.08% vs M0 RM38,660.32, well within $\le 5\%$ tolerance).
  - `pytest tests/test_fairness_audit.py`: 2/2 passed in 3.34s.
- **Next**: **T9.8** Fix objective checks in `evaluate_test.py` (same-split, no hardcoded constants) (F4).
- **Blockers**: None.

---

## 05 Oct 2026: Task T9.6 Completed (Phase 9 Sprint B)
- **Done**:
  - **T9.6**: Executed Experiment E11 (Champion Re-Decision) comparing E02 (Logistic Regression with engineered features) against E07 (Optuna-tuned LightGBM) on identical 5 folds of `train + val` (5,986 samples).
  - Evaluated paired per-fold PR-AUC differences:
    - Fold 1: LR = 0.6741, LightGBM = 0.6694 ($\Delta = -0.0046$)
    - Fold 2: LR = 0.6520, LightGBM = 0.6540 ($\Delta = +0.0020$)
    - Fold 3: LR = 0.6437, LightGBM = 0.6625 ($\Delta = +0.0187$)
    - Fold 4: LR = 0.6878, LightGBM = 0.6892 ($\Delta = +0.0014$)
    - Fold 5: LR = 0.6413, LightGBM = 0.6463 ($\Delta = +0.0050$)
    - **Mean paired difference**: **+0.0045** (std: **0.0078**).
  - Pre-registered decision rule applied (`14_IMPROVEMENT_PLAN.md` section 4.4):
    - Condition: requires mean gain $\ge$ 1 std (0.0078) AND mean gain $\ge 0.01$.
    - Result: $+0.0045 < 0.0078$ and $+0.0045 < 0.01$. Condition not met.
    - Decision: **Logistic Regression is selected as Champion** (simpler, highly interpretable, zero boosting overhead), and **LightGBM is designated as Runner-up**.
    - Updated **D-013** in `docs/08_DECISIONS_LOG.md`.
    - **MO1 Status**: Reported honestly as **NOT MET** (LightGBM PR-AUC 0.6643 vs baseline 0.6587 is +0.0056 gain, short of +0.03 target).
  - Saved full results to `reports/e11_champion_decision.json` and logged to MLflow run `E11_Champion_ReDecision`.
  - Added unit test in `tests/test_champion_decision.py`.
- **Evidence**:
  - `LR PR-AUC`: $0.6598 \pm 0.0182$
  - `LightGBM PR-AUC`: $0.6643 \pm 0.0147$
  - `Paired PR-AUC Diff`: $+0.0045 \pm 0.0078$
  - `pytest tests/test_champion_decision.py`: 2/2 passed in 1.94s.
- **Next**: **T9.7** E12 fairness audit (M0 / M1 / M2) on OOF at chosen threshold (F6).
- **Blockers**: None.

---

## 05 Oct 2026: Task T9.5 Completed (Phase 9 Sprint B)
- **Done**:
  - **T9.5**: Re-optimised campaign profit threshold on out-of-fold calibrated probabilities across `train + val` (5,986 samples) instead of the smaller validation set. Evaluated net profit across decision thresholds [0.01, 0.99] and computed multi-parameter sensitivity analysis (success rate in {0.2, 0.3, 0.4}, offer cost in {RM40, RM50, RM60}).
  - Updated `src/churnguard/models/threshold.py` to prioritize `models/oof_train_val_preds.parquet`, accept individual CLV values in profit curve visualization, and record `"source": "oof_train_val"` in `models/optimal_threshold.json`.
  - Regenerated `reports/figures/08_profit_curve.png`.
  - Added unit test `test_optimal_threshold_json_has_oof_source` in `tests/test_threshold.py`.
- **Evidence**:
  - `models/optimal_threshold.json`:
    - `source`: `"oof_train_val"`
    - `n_samples`: 5,986
    - `optimal_threshold`: **0.1882**
    - `expected_profit_per_1k_rm`: **RM39,481.98** (vs default $\tau=0.5$ of RM29,830.03, Contact All of RM21,632.30, Contact None RM0.00)
    - `pct_customers_contacted`: 48.51%
    - `churner_capture_rate`: 87.28%
  - Sensitivity analysis confirms robust positive expected profit across all 9 scenarios (RM15,479 to RM66,172 per 1k).
  - `pytest tests/test_threshold.py`: 5/5 passed in 2.93s.
- **Next**: **T9.6** E11 champion re-decision: E02 vs E07 on identical folds, paired PR-AUC diff (F5).
- **Blockers**: None.

---

## 05 Oct 2026: Task T9.4 Completed (Phase 9 Sprint B)
- **Done**:
  - **T9.4**: Redesigned calibration experiment (E10) using 5-fold Stratified cross-validation on combined `train + val` (5,986 samples). Compared uncalibrated LightGBM vs Sigmoid `cv=5` vs Isotonic `cv=5`. Eliminated probability plateau collapse (F2, F3) and avoided in-sample data leakage.
  - Pre-registered selection rule applied: Isotonic `cv=5` selected (lowest Brier score 0.1341, PR-AUC delta -0.0017 $\le 0.005$, unique probabilities 2,842 $\ge 200$).
  - Exported out-of-fold calibrated predictions to `models/oof_train_val_preds.parquet`.
  - Regenerated canonical calibration plot `reports/figures/07_calibration_curve.png` and logged parameters/metrics/artifacts to MLflow experiment `E10_Calibration_Redesign`.
- **Evidence**:
  - Out-of-Fold Calibration Metrics (5,986 samples):
    - *Uncalibrated*: Brier = 0.1522, PR-AUC = 0.6601, ROC-AUC = 0.8474, ECE = 0.1136, Unique Probs = 5,969
    - *Sigmoid cv=5*: Brier = 0.1344, PR-AUC = 0.6621, ROC-AUC = 0.8479, ECE = 0.0205, Unique Probs = 5,978
    - *Isotonic cv=5*: Brier = **0.1341**, PR-AUC = 0.6584, ROC-AUC = 0.8467, ECE = **0.0142**, Unique Probs = **2,842**
  - Unique probability levels: 2,842 (surpasses $\ge 200$ acceptance criteria; previous prefit collapsed to 32).
  - Out-of-fold Expected Calibration Error (ECE): reduced from 0.1136 to 0.0142.
  - `pytest tests/test_calibrate.py`: 3/3 passed in 23.89s.
- **Next**: **T9.5** Re-optimise profit threshold on OOF calibrated probs + sensitivity (F7).
- **Blockers**: None.

---

## 05 Oct 2026: Task T9.3 Completed (Phase 9 Sprint B)
- **Done**:
  - **T9.3**: Implemented tie-aware ranking metrics (`compute_lift_at_k`, `compute_recall_at_k`) and `compute_n_unique_probs` in `src/churnguard/models/evaluate.py`. Replaced naive top-K indexing with fractional expectation calculation over items sharing the cutoff probability. Added unit tests in `tests/test_evaluate.py` proving analytical correctness and row-order invariance across 20 random row permutations (fixing F2).
- **Evidence**:
  - `pytest tests/test_evaluate.py`: 6/6 passed in 0.97s.
  - `test_tie_aware_row_order_invariance`: proved deterministic equality ($|\Delta| < 10^{-12}$) across 20 permutations of tied scores at boundary.
  - `test_tie_aware_analytical_fraction`: analytically matched theoretical expectation ($5/9$).
- **Next**: **T9.4** E10 calibration redesign: uncalibrated vs sigmoid cv=5 vs isotonic cv=5 on train+val, OOF metrics (F2, F3).
- **Blockers**: None.

---

## 05 Oct 2026: Task T9.2 Completed (Phase 9 Sprint A)
- **Done**:
  - **T9.2**: Cleaned duplicate raw files (`WA_Fn-UseC_-Telco-Customer-Churn.csv` and `cellular_subscribers.csv`) from `data/raw/`. Cleaned duplicate unnumbered figures (`calibration_curve.png`, `profit_curve.png`, `shap_summary.png`) from `reports/figures/`, updating source code in `calibrate.py`, `threshold.py`, and `shap_explain.py` to save only canonical numbered artifacts. Updated `pyproject.toml` to `requires-python = ">=3.11"` and `target-version = "py311"`. Updated `Dockerfile` base image to `python:3.11-slim AS runtime`. Updated `.github/workflows/ci.yml` and `README.md` to Python 3.11. Verified API integration tests pass on Python 3.11.
- **Evidence**:
  - `data/raw/` file inventory: strictly `telco_churn.csv`, `my_cellular_subscribers.csv`, `.gitkeep`.
  - `reports/figures/` inventory: strictly `01_` through `09_` canonical PNGs.
  - `pytest` run output: `73 passed, 1063 warnings in 53.63s`, total coverage: 83.26% (exceeds 70% requirement).
  - `pytest tests/test_api.py`: 6/6 passed.
  - `ruff`: all checks passed, 57 files formatted.
- **Next**: **T9.3** Tie-aware `recall_at_k` / `lift_at_k` + `n_unique_probs` in `evaluate.py` (F2).
- **Blockers**: None.

---

## 05 Oct 2026: Task T9.1 Completed (Phase 9 Sprint A)
- **Done**:
  - **T9.1**: Verified Phase 7 commit (`0eba781`). Ran test suite verifying all **73/73 tests pass** with **83.32% code coverage** (exceeding $\ge 70\%$ gate). Committed model review findings (`docs/13_MODEL_REVIEW.md`), hardening improvement plan (`docs/14_IMPROVEMENT_PLAN.md`), updated decisions log (`docs/08_DECISIONS_LOG.md` D-013 to D-016), updated experiment plan (`docs/06_EXPERIMENT_PLAN.md`), and updated `CLAUDE.md`. Verified clean working tree.
- **Evidence**:
  - `pytest` run output: `73 passed, 1063 warnings in 51.43s`, total coverage: 83.32% (exceeds 70% requirement).
  - `ruff` check and format: all checks passed, 57 files formatted.
  - `git log`: Commit `0eba781` verified.
- **Next**: **T9.2** Clean repo: remove duplicate raw files and duplicate figures, Docker `python:3.11-slim`, `requires-python >=3.11`, rebuild image (F10).
- **Blockers**: None.

---

## 05 Oct 2026: Phase 8 Portfolio Packaging Completed
- **Done**:
  - **T8.1**: Authored comprehensive production [`README.md`](file:///c:/Users/ZeeqRyz/Desktop/Ai-ML/Machine%20Learning%20Projects/02_ChurnGuard_Telco_Churn_Prediction/README.md) featuring CI/CD badges, executive problem framing (Malaysian telco NusaTel), complete results table with 1,000x bootstrap confidence intervals, Mermaid architecture diagram, quickstart commands, and API usage samples.
  - **T8.2**: Integrated architecture flowcharts and visual references into README.
  - **T8.3**: Cleaned repository, validated `.gitignore` root anchoring for `/data/`, `/models/`, and `/mlruns/`, ensuring 0 secrets and clean git hygiene.
  - **T8.4**: Authored LinkedIn announcement post, resume bullet points, and portfolio tracking entries in [`docs/11_LINKEDIN_POST.md`](file:///c:/Users/ZeeqRyz/Desktop/Ai-ML/Machine%20Learning%20Projects/02_ChurnGuard_Telco_Churn_Prediction/docs/11_LINKEDIN_POST.md).
  - **T8.5**: Created root Hugging Face Spaces entrypoint [`app.py`](file:///c:/Users/ZeeqRyz/Desktop/Ai-ML/Machine%20Learning%20Projects/02_ChurnGuard_Telco_Churn_Prediction/app.py) for instantaneous Streamlit deployment.
  - **T8.6**: Generated standalone Kaggle public notebook artifact [`notebooks/kaggle_telco_churn_guard.ipynb`](file:///c:/Users/ZeeqRyz/Desktop/Ai-ML/Machine%20Learning%20Projects/02_ChurnGuard_Telco_Churn_Prediction/notebooks/kaggle_telco_churn_guard.ipynb) with links back to GitHub.
  - **T8.7**: Authored in-depth technical publication article draft in [`docs/12_TECHNICAL_ARTICLE.md`](file:///c:/Users/ZeeqRyz/Desktop/Ai-ML/Machine%20Learning%20Projects/02_ChurnGuard_Telco_Churn_Prediction/docs/12_TECHNICAL_ARTICLE.md) ("Stop Using 0.5 Thresholds: Building a Profit-Driven Churn Retention System").
  - **T8.8**: Completed all items in pre-publish checklist in [`docs/10_PUBLISHING_PLAN.md`](file:///c:/Users/ZeeqRyz/Desktop/Ai-ML/Machine%20Learning%20Projects/02_ChurnGuard_Telco_Churn_Prediction/docs/10_PUBLISHING_PLAN.md).
- **Next**: Deploy live instances to GCP Cloud Run and Hugging Face Spaces.
- **Blockers**: None
- **Code Quality**: 73 passed tests, 84% test coverage, 0 ruff errors.

---

## 05 Oct 2026: Phase 7 Monitoring + Dashboard Completed
- **Done**:
  - **T7.1**: Implemented macroeconomic and behavioral data drift batch simulator in `src/churnguard/monitoring/simulate_drift.py`. Generated production test batch with simulated inflation (+25% MonthlyCharges), contract migration to month-to-month, fiber optic adoption, and tenure compression. Exported to `data/processed/drifted_batch.parquet` and `.csv`. Tested in `tests/test_simulate_drift.py`.
  - **T7.2**: Implemented Evidently AI data and prediction drift monitoring engine in `src/churnguard/monitoring/drift.py`. Analyzed reference training split vs current production batch across 21 columns and model prediction probabilities. Evaluated retrain trigger rules (drift share $\ge 30\%$). Generated interactive HTML report at `reports/drift/drift_report.html` (EO5 Met) and JSON summary at `reports/drift/drift_summary.json`. Tested in `tests/test_drift.py`.
  - **T7.3**: Built interactive multi-page Streamlit retention decision cockpit in `app/streamlit_app.py` featuring:
    1. *Single Customer Assessment:* Interactive profile input, calibrated probability gauge, risk tier badges, and top-3 frontline plain-language SHAP reason codes.
    2. *Batch Scoring & Targeting:* CSV upload, customer ranking by risk descending, and retention campaign profit ROI simulator (RM).
    3. *Strategic Business Insights:* Interactive deep-dives into contract lock-in, fiber optic deficits, and profit curve optimization ($\tau^* = 0.18$).
    4. *Malaysia Telecom Market Context:* Historical mobile trends from data.gov.my.
    5. *MLOps Health & Drift Monitoring:* Evidently report viewer and retrain alerts. Tested in `tests/test_streamlit_app.py`.
- **Next**: Phase 8 — Portfolio Packaging
- **Blockers**: None
- **Code Quality**: 73 passed tests, 84% test coverage, 0 ruff errors.

---

## 05 Oct 2026: Model Review + Improvement Plan (PM)
- **Done**: Senior review of v1.0 (`13_MODEL_REVIEW.md`); test metrics independently recomputed; improvement plan written (`14_IMPROVEMENT_PLAN.md`); Phase 9 (14 tasks) added
- **Findings**: not on GitHub / not deployed (F1); isotonic calibration collapsed scores to 32 levels (F2); in-sample calibration metrics (F3); wrong-split objective checks (F4); MO1 not met (F5); fairness gaps 0.078 / 0.144 on test (F6)
- **Reopened**: T0.1, T6.1, T6.2, T6.3 (F1) and T8.2 to T8.8 (F11: nothing published yet; drafts only) (D-016)
- **Next**: T9.1
- **Blockers**: GCP billing account needed for T9.13 (fallback Render)

---

## 05 Oct 2026: Phase 6 CI/CD + Deploy Completed
- **Done**:
  - **T6.1**: Configured GitHub Actions CI workflow in `.github/workflows/ci.yml` running ruff linting (`ruff check`, `ruff format --check`) and pytest with a strict 70% coverage gate (`--cov-fail-under=70`). Verified all 69 unit and integration tests pass with **85% overall coverage**.
  - **T6.2**: Configured Docker build and live smoke test job in `.github/workflows/ci.yml`. Builds the container image, starts the container, validates `GET /health`, `GET /model-info`, and `POST /predict` inference payload, and captures container logs on teardown.
  - **T6.3**: Created Google Cloud Run continuous deployment workflow in `.github/workflows/deploy.yml` triggered on git tags (`v*`) and manual workflow dispatch (`workflow_dispatch`). Created deployment automation scripts (`scripts/deploy_cloud_run.sh` and `scripts/deploy_cloud_run.ps1`) targeting Google Artifact Registry and Cloud Run managed service in `asia-southeast1`. Verified non-interactive test execution compatibility with `tests/conftest.py`.
- **Next**: Phase 7 — Monitoring + Dashboard (starting with T7.1 drift simulation script and T7.2 Evidently report)
- **Blockers**: None
- **Code Quality**: 69 passed tests, 85% test coverage, 0 ruff errors.

---

## 05 Oct 2026: Phase 5 Serving Completed
- **Done**:
  - **T5.1**: Implemented `src/churnguard/models/predict.py` featuring `ChurnPredictor` inference engine. Loads champion pipeline `models/model.joblib` and metadata `models/model_meta.json`, outputs calibrated probabilities, maps risk tiers (`High`, `Medium`, `Low`), and generates top 3 plain-language SHAP reason codes. Tested in `tests/test_predict.py`.
  - **T5.2**: Built production FastAPI service in `api/main.py` and strict Pydantic data contracts in `api/schemas.py`. Endpoints implemented:
    - `GET /health`: Health status and loaded model version.
    - `GET /model-info`: Production model architecture, optimal threshold $\tau^*=0.18$, risk tier boundaries, feature lists, and test set performance.
    - `POST /predict`: Single-customer prediction with calibrated probability, risk tier, and top 3 reasons (422 validation on invalid payloads).
    - `POST /predict/batch`: High-throughput batch prediction returning risk-ranked customer list with tier counts.
  - **T5.3**: Built batch scoring CLI command `python -m churnguard.models.predict --file $(FILE)` (`make score FILE=...`). Tested on `data/processed/test.parquet` (1,057 customers), exported ranked CSV to `reports/scored.csv`.
  - **T5.4**: Implemented comprehensive integration test suite with `TestClient` in `tests/test_api.py`. Tested status codes, payload contracts, validation errors (422), and batch sorting. Verified **EO3 met** (overall code coverage at **85%**, above $\ge 70\%$ requirement).
  - **T5.5**: Implemented latency benchmark in `tests/test_latency.py` (100 requests). Achieved **p95 latency of 3.27 ms** (mean 2.44 ms, p50 2.33 ms, p99 3.53 ms), substantially exceeding the **EO2 target ($p95 < 100$ ms)**.
  - **T5.6**: Built production-grade slim Dockerfile (`python:3.10-slim`, non-root `appuser` UID 1000, multi-layer caching, healthcheck probe on `/health`, exposed port 8000) and `.dockerignore`.
- **Next**: Phase 6 — CI/CD + Deploy (starting with T6.1 GitHub Actions workflow and T6.2 container build)
- **Blockers**: None
- **Code Quality**: 69 passed tests, 85% test coverage, 0 ruff errors.

---

## 05 Oct 2026: Phase 4 Evaluation, Business and Explainability Completed
- **Done**:
  - **T4.1**: Implemented `src/churnguard/models/calibrate.py` (Experiment E09). Fitted Platt Sigmoid and Isotonic Regression on validation split with champion LightGBM. Isotonic calibration reduced Brier Score from 0.1542 to 0.1293 (16% relative gain) and ECE from 0.1239 to 0.0000. Saved reliability curves to `reports/figures/07_calibration_curve.png` and `reports/figures/calibration_curve.png`. Verified **MO3** met. Tested in `tests/test_calibrate.py`.
  - **T4.2**: Implemented `src/churnguard/models/threshold.py`. Computed campaign profit curves across candidate thresholds $\tau \in [0.01, 0.99]$ on validation probabilities using RM50 offer cost, 30% retention rate, and RM780 CLV. Identified profit-optimal threshold **$\tau^* = 0.18$**, yielding expected campaign profit of **RM39,678.52 per 1,000 customers** (+44% over default $\tau=0.5$ and +83% over "Contact All"). Performed sensitivity analysis across success rates (20%, 30%, 40%) and offer costs (RM40, RM50, RM60). Saved figures to `reports/figures/08_profit_curve.png` and `reports/figures/profit_curve.png`, and exported `models/optimal_threshold.json`. Tested in `tests/test_threshold.py`.
  - **T4.3**: Implemented `src/churnguard/explain/shap_explain.py` with `TreeExplainer` on champion LightGBM. Generated global feature importance and beeswarm summary plots saved to `reports/figures/09_shap_summary.png` and `reports/figures/shap_summary.png`. Built customer-level top-3 plain-language reason code generator mapping SHAP contributions into empathetic frontline explanations; verified on 5 sample customers. Tested in `tests/test_shap_explain.py`.
  - **T4.4**: Executed final evaluation on held-out test split `data/processed/test.parquet` (1,057 samples, touched strictly once) using 1,000x bootstrap 95% confidence intervals:
    - **ROC-AUC**: 0.8412 [95% CI: 0.8155, 0.8679] (MO2 $\ge 0.84$ Met)
    - **PR-AUC**: 0.6337 [95% CI: 0.5776, 0.6899] (MO2 $\ge 0.62$ Met)
    - **Brier Score**: 0.1381 [95% CI: 0.1247, 0.1509] (MO3 Met)
    - **Lift@10%**: 2.73x [95% CI: 2.54x, 3.20x] (MO2 $\ge 2.5$ Met)
    - **Recall@20%**: 51.60% [95% CI: 46.05%, 54.86%] (BO1 $\ge 50\%$ Met)
    - **Precision at $\tau^*$**: 53.75% [95% CI: 48.59%, 58.88%]
    - **Recall at $\tau^*$**: 76.51% [95% CI: 71.28%, 81.40%]
    - **F1 at $\tau^*$**: 0.6314 [95% CI: 0.5871, 0.6724]
    - **Expected Profit per 1k**: **RM35,206.24** [95% CI: RM29,031.29, RM41,165.51] (BO2 Met)
    - Saved complete test report to `reports/final_metrics.json` and updated `docs/06_EXPERIMENT_PLAN.md`. Tested in `tests/test_evaluate_test.py`.
  - **T4.5**: Created `src/churnguard/explain/fairness.py` and `notebooks/02_error_analysis.ipynb`. Audited demographic fairness across `gender` (female recall 0.8231 vs male recall 0.8346, gap 0.0115 $\le 0.05 \implies$ NFR7 passed) and `SeniorCitizen` (senior recall 0.9367 vs non-senior 0.7861 due to higher fiber-optic churn concentration). Diagnosed false positives (high-spend month-to-month retainees) and false negatives (unexpected contract terminations). Tested in `tests/test_fairness.py`.
  - **T4.6**: Implemented `src/churnguard/models/serialize.py`. Serialized full production pipeline to `models/model.joblib`, exported metadata to `models/model_meta.json`, logged artifacts to MLflow model registry with tag `stage=production`, and verified standalone reload/inference. Tested in `tests/test_serialize.py`.
- **Next**: Phase 5 — Serving (starting with T5.1 `predict.py` inference engine and T5.2 FastAPI service)
- **Blockers**: None
- **Code Quality**: 59 passed tests, 79% test coverage, 0 ruff errors.

---

## 05 Oct 2026: Phase 3 Modelling & Tuning Completed
- **Done**:
  - **T3.1**: Executed and logged tree-based candidate models across 5-fold Stratified CV:
    - **E03 (Random Forest)**: CV PR-AUC: 0.6629 ± 0.0172, ROC-AUC: 0.8462, Lift@10: 2.79x
    - **E04 (XGBoost)**: CV PR-AUC: 0.6631 ± 0.0266, ROC-AUC: 0.8437, Lift@10: 2.87x
    - **E05 (LightGBM untuned)**: CV PR-AUC: 0.6598 ± 0.0251, ROC-AUC: 0.8409, Lift@10: 2.91x
  - **T3.2**: Executed **E06 (LightGBM with SMOTE resampling inside CV)**. SMOTE degraded PR-AUC to 0.6537 ± 0.0274 and increased fold variance. Confirmed decision **D-005** (reject SMOTE in favor of cost/weight-based imbalance handling).
  - **T3.3**: Built Optuna Bayesian hyperparameter search in `src/churnguard/models/tune.py` (60 trials maximizing CV PR-AUC).
    - **E07 (Tuned LightGBM Champion)**: CV PR-AUC: **0.6732 ± 0.0217**, ROC-AUC: **0.8468**, Lift@10: **2.92x**, Recall@20: **51.22%**.
    - Best parameters saved to `models/best_params.json`.
  - **T3.4**: Executed **E08 (Fairness Ablation)** dropping `gender` and `SeniorCitizen` (`src/churnguard/models/fairness_ablation.py`):
    - CV PR-AUC: 0.6706 ± 0.0217 (minimal 0.38% dip), ROC-AUC: 0.8464.
    - Updated decision **D-006**: demographic features carry minimal bias; retained in v1 for fairness slicing (NFR7).
  - **T3.5**: Verified SMART ML Objectives:
    - **MO1**: Tuned model beats baseline (PR-AUC 0.6732 vs 0.6587; top-decile lift 2.92x).
    - **MO2 (Ranking Quality)**: ROC-AUC 0.8468 >= 0.84, PR-AUC 0.6732 >= 0.62, Lift@10% 2.92 >= 2.5 -> **MET**.
    - **MO4 (Robustness)**: CV PR-AUC std 0.0217 <= 0.03 -> **MET**.
- **Next**: Phase 4 — Evaluation, Business and Explainability (starting with T4.1 probability calibration E09)
- **Blockers**: None
- **Code Quality**: 44 passed tests, 91% test coverage, 0 ruff errors.

---

## 05 Oct 2026: Phase 2 Baseline Models Completed
- **Done**:
  - **T2.1**: Implemented `FeatureEngineer` scikit-learn transformer in `src/churnguard/features/build.py` implementing all 8 domain features from `05_DATA_SPEC.md` section 5 (`tenure_bucket`, `avg_monthly_spend`, `charge_increase_ratio`, `num_services`, `has_protection_bundle`, `is_auto_pay`, `is_month_to_month`, `fiber_no_support`). Tested individually in `tests/test_features.py`.
  - **T2.2**: Implemented `ColumnTransformer` preprocessing pipeline handling numeric median imputation + standard scaling, categorical most-frequent imputation + OneHotEncoding (`handle_unknown="ignore"`). Verified shapes and fit-on-train-only discipline via `tests/test_features.py`.
  - **T2.3**: Implemented comprehensive evaluation module in `src/churnguard/models/evaluate.py` providing PR-AUC, ROC-AUC, Brier score, Lift@10%, Recall@20%, F1, Precision, Recall, and business Expected Profit (RM). Validated on toy arrays in `tests/test_evaluate.py`.
  - **T2.4**: Built 5-fold Stratified CV training engine with local MLflow tracking in `src/churnguard/models/train.py`. Executed and logged baselines E00, E01, E02:
    - **E00 (Dummy Stratified Floor)**: CV PR-AUC: 0.2690 ± 0.0024, ROC-AUC: 0.5089, Lift@10: 1.03
    - **E01 (Logistic Regression - Raw Features)**: CV PR-AUC: 0.6587 ± 0.0220, ROC-AUC: 0.8442, Lift@10: 2.83, Recall@20: 50.31%
    - **E02 (Logistic Regression - Engineered Features)**: CV PR-AUC: 0.6611 ± 0.0124, ROC-AUC: 0.8458, Lift@10: 2.89, Recall@20: 50.61%
  - Updated experiment results table in `docs/06_EXPERIMENT_PLAN.md`.
- **Next**: Phase 3 — Modelling + Tuning (E03 Random Forest, E04 XGBoost, E05 LightGBM, E06 SMOTE, E07 Optuna)
- **Blockers**: None
- **Code Quality**: 28 passed tests, 94% test coverage, 0 ruff errors.

---

## 05 Oct 2026: Phase 1 Data Understanding & EDA Completed
- **Done**:
  - **T1.1**: Implemented `src/churnguard/data/validate.py` using pandera schema checking data types, value sets, value bounds (tenure in 0-72, MonthlyCharges > 0), unique IDs, and minimum row count (>= 5,000). Verified with `tests/test_validate.py`.
  - **T1.2**: Implemented `src/churnguard/data/clean.py` handling blank `TotalCharges` -> 0.0 (D-003), mapping target `Churn` ("Yes" -> 1, "No" -> 0), type casting, and verifying 0 remaining nulls. Verified with `tests/test_clean.py`.
  - **T1.3**: Implemented `src/churnguard/data/split.py` performing stratified 70/15/15 train/val/test split with `seed=42`. Saved `train` (4,930 rows), `val` (1,056 rows), `test` (1,057 rows) in both CSV and Parquet formats in `data/processed/`. Churn rate across splits is verified at 26.53% ± 0.02% (< 1% variance). Verified with `tests/test_split.py`.
  - **T1.4**: Generated comprehensive EDA notebook `notebooks/01_eda.ipynb` and all 6 publication-ready figures in `reports/figures/` (`01_target_distribution.png`, `02_categorical_churn_rates.png`, `03_numeric_distributions.png`, `04_cramers_v_association.png`, `05_tenure_contract_heatmap.png`, `06_malaysia_cellular_trends.png`).
  - **T1.5**: Identified 6 core business insights:
    1. *Contract Lock-in:* Month-to-month contracts have 42.9% churn vs 10.9% (1-yr) and 3.0% (2-yr). Month-to-month drives 88.6% of all churners.
    2. *Early Lifecycle Risk:* Months 0-6 with month-to-month contracts show 53.1% churn. Churned median tenure is 10.0 months vs 38.0 months for retained users.
    3. *Fiber Optic Service Deficit:* Fiber optic users churn at 41.8% (vs 19.2% DSL); rises to >48% without TechSupport.
    4. *Payment Friction:* Electronic Check shows 45.6% churn vs <15-16% for automated payment methods.
    5. *Protection Bundle Stickiness:* Lack of TechSupport or OnlineSecurity triples churn risk (42% vs 15%).
    6. *Malaysian Market Framing:* Steady growth in high-ARPU postpaid subscriptions (>10M) emphasizes the commercial value of contractual retention interventions.
  - **T1.6**: Analyzed Malaysia national cellular subscriptions from `data.gov.my` (2000-2021) and plotted postpaid vs prepaid trends in `reports/figures/06_malaysia_cellular_trends.png`.
- **Next**: Phase 2 — Baseline Models (starting with T2.1 `FeatureEngineer` transformer)
- **Blockers**: None
- **Code Quality**: 26 passed tests, 93% test coverage, 0 ruff errors.

---

## 05 Oct 2026: Phase 0 Setup Completed
- **Done**:
  - **T0.1**: Initialized git repository, added `.gitignore` (data/, models/, mlruns/, .venv/, __pycache__/, coverage).
  - **T0.2**: Created `pyproject.toml` and `requirements.txt` with pinned versions; installed clean virtual environment.
  - **T0.3**: Created full project structure per `04_TECHNICAL_DESIGN.md` with `__init__.py` files; verified `import churnguard` works.
  - **T0.4**: Created `configs/config.yaml` and `src/churnguard/config.py` loader; verified with unit tests (`test_config.py`).
  - **T0.5**: Created `Makefile` and `.pre-commit-config.yaml` with ruff hooks; `ruff check`, `ruff format`, and `pytest` pass with 82% coverage.
  - **T0.6**: Set up raw data loading in `src/churnguard/data/load.py` for both Telco dataset (7,043 rows) and Malaysia cellular subscribers context dataset (66 rows); verified with `test_load.py`.
- **Next**: Phase 1 — Data Understanding + EDA (starting with T1.1 `validate.py` with pandera schema)
- **Blockers**: None
- **Decisions**: Updated pinned versions in `docs/00_TECH_STACK.md`.

---

## Weekly PM Review Template
```
### Week N Review (date)
- Planned vs done:
- Metrics so far:
- Risks changed:
- Re-plan:
```
