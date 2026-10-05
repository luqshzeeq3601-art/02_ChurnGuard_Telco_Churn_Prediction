# 09. Progress Log

> Newest entry on top. Each session: Done, Next, Blockers. Weekly PM review every Sunday.

## Current Status
| Item | Value |
|---|---|
| Current phase | Phase 6: CI/CD + Deploy (W5) |
| Next task | **T6.1** GitHub Actions: lint + test + coverage gate |
| Overall progress | 33 / 47 tasks |
| Health | On track |

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
