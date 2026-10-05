# 09. Progress Log

> Newest entry on top. Each session: Done, Next, Blockers. Weekly PM review every Sunday.

## Current Status
| Item | Value |
|---|---|
| Current phase | Phase 3: Modelling + Tuning (W2-W3) |
| Next task | **T3.1** Run E03 (Random Forest), E04 (XGBoost), E05 (LightGBM) |
| Overall progress | 16 / 47 tasks |
| Health | On track |

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
