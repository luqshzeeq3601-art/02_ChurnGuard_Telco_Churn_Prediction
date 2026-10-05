# 07. Tasks Backlog

> Tick `[x]` when acceptance criteria are met. Work top to bottom within a phase.

## Phase 0: Setup (W1)
- [x] **T0.1** Init git repo, `.gitignore` (data/, models/, mlruns/, .venv/, __pycache__/) | AC: first commit pushed to GitHub
- [x] **T0.2** `pyproject.toml` + `requirements.txt` (pinned versions) | AC: fresh venv installs cleanly
- [x] **T0.3** Create folder structure per `04_TECHNICAL_DESIGN.md` section 3 with `__init__.py` | AC: `import churnguard` works
- [x] **T0.4** `configs/config.yaml` (paths, seed, split ratios, cost assumptions) + `config.py` loader | AC: unit test loads config
- [x] **T0.5** `Makefile` with `setup`, `lint`, `test` targets; pre-commit with ruff | AC: `make lint test` passes
- [x] **T0.6** Download dataset to `data/raw/telco_churn.csv` (`src/churnguard/data/load.py`) | AC: file present, 7,043 rows

## Phase 1: Data Understanding + EDA (W1)
- [ ] **T1.1** `validate.py` with pandera schema from `05_DATA_SPEC.md` section 4 | AC: passes on raw data; tests cover a bad-value case
- [ ] **T1.2** Cleaning function (TotalCharges fix, target to 0/1) | AC: no nulls; test covers blank TotalCharges
- [ ] **T1.3** `split.py` stratified 70/15/15 saved to `data/processed/` | AC: churn rate within plus or minus 1% across splits
- [ ] **T1.4** EDA notebook following `05_DATA_SPEC.md` section 6 | AC: all checklist items done, figures in `reports/figures/`
- [ ] **T1.5** Write 5+ business insights | AC: in notebook + progress log
- [ ] **T1.6** Malaysia context: load data.gov.my subscribers CSV, plot postpaid vs prepaid trend | AC: chart in `reports/figures/`, used in README

## Phase 2: Baseline (W2)
- [ ] **T2.1** `FeatureEngineer` transformer (features in `05_DATA_SPEC.md` section 5) | AC: unit tests per feature
- [ ] **T2.2** `ColumnTransformer` preprocessing in `features/build.py` | AC: fit on train only; output shape test
- [ ] **T2.3** `evaluate.py` metrics (PR-AUC, ROC-AUC, Brier, Lift@10, Recall@20) | AC: tested on toy arrays
- [ ] **T2.4** `train.py` with CV + MLflow logging | AC: E00, E01, E02 logged; results table updated

## Phase 3: Modelling + Tuning (W2-W3)
- [ ] **T3.1** Run E03, E04, E05 | AC: logged; results table updated
- [ ] **T3.2** Run E06 (SMOTE inside CV via imblearn Pipeline) | AC: logged; decision recorded
- [ ] **T3.3** `tune.py` Optuna (E07) | AC: best params saved; MO1 met
- [ ] **T3.4** E08 fairness ablation | AC: decision D-006 updated
- [ ] **T3.5** Check MO2, MO4 on CV | AC: met or reason logged

## Phase 4: Evaluation, Business and Explainability (W3-W4)
- [ ] **T4.1** Calibration on val (E09) | AC: Brier improves; reliability plot saved (MO3)
- [ ] **T4.2** `threshold.py` profit curve + optimal threshold + sensitivity | AC: plot saved; threshold in `model_meta.json`
- [ ] **T4.3** `shap_explain.py`: global summary + per-customer top 3 reasons with plain-language templates | AC: reasons readable for 5 sample customers
- [ ] **T4.4** Final test evaluation with bootstrap CI | AC: `reports/final_metrics.json`; results table filled
- [ ] **T4.5** Error analysis + fairness metrics by gender / SeniorCitizen | AC: notebook 02 done; NFR7 checked
- [ ] **T4.6** Save final pipeline `models/model.joblib` + `model_meta.json`; register in MLflow | AC: reload and predict test passes

## Phase 5: Serving (W4)
- [ ] **T5.1** `predict.py` (load model, predict, tier, reasons) | AC: unit tests
- [ ] **T5.2** FastAPI `api/main.py` + `schemas.py` (endpoints per design section 7) | AC: `/docs` works; invalid input returns 422
- [ ] **T5.3** Batch CLI `make score FILE=...` | AC: ranked CSV output
- [ ] **T5.4** API tests with TestClient | AC: coverage at least 70% overall (EO3)
- [ ] **T5.5** Latency check (100 requests) | AC: p95 under 100 ms (EO2)
- [ ] **T5.6** Dockerfile (slim, non-root) + `make docker-build docker-run` | AC: container serves `/health`

## Phase 6: CI/CD + Deploy (W5)
- [ ] **T6.1** GitHub Actions: lint + test + coverage gate | AC: green badge in README
- [ ] **T6.2** Docker build + smoke test job | AC: passes on main
- [ ] **T6.3** Deploy to GCP Cloud Run (manual first, then tag-triggered) | AC: public `/docs` URL works (EO4)

## Phase 7: Monitoring + Dashboard (W5-W6)
- [ ] **T7.1** Simulate drifted batch | AC: script documented
- [ ] **T7.2** `drift.py` Evidently report | AC: HTML in `reports/drift/` (EO5)
- [ ] **T7.3** Streamlit dashboard (needed for public demo) | AC: upload CSV or form input, ranked list, reasons, segment charts

## Phase 8: Portfolio Packaging (W6)
- [ ] **T8.1** README: problem, results table, architecture diagram, how to run, limitations | AC: complete
- [ ] **T8.2** Demo GIF / screenshots of API + dashboard | AC: in README
- [ ] **T8.3** Clean repo, make public, pin repo on GitHub | AC: done
- [ ] **T8.4** LinkedIn post + add to CV and portfolio spreadsheet | AC: posted; tracker status = Done
- [ ] **T8.5** Deploy Streamlit demo to Hugging Face Spaces | AC: public Space URL loads and scores a sample
- [ ] **T8.6** Publish Kaggle notebook (EDA + model summary, link to GitHub) | AC: public notebook URL
- [ ] **T8.7** (Optional) Medium / dev.to write-up | AC: published, linked in README
- [ ] **T8.8** Run pre-publish checklist in `10_PUBLISHING_PLAN.md` section 3 | AC: all items ticked
