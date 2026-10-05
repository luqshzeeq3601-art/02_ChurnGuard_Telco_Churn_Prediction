# 07. Tasks Backlog

> Tick `[x]` when acceptance criteria are met. Work top to bottom within a phase.

## Phase 0: Setup (W1)
- [ ] **T0.1** Init git repo, `.gitignore` (data/, models/, mlruns/, .venv/, __pycache__/) | AC: first commit pushed to GitHub | **REOPENED 05 Oct (F1, see 13_MODEL_REVIEW.md)**
- [x] **T0.2** `pyproject.toml` + `requirements.txt` (pinned versions) | AC: fresh venv installs cleanly
- [x] **T0.3** Create folder structure per `04_TECHNICAL_DESIGN.md` section 3 with `__init__.py` | AC: `import churnguard` works
- [x] **T0.4** `configs/config.yaml` (paths, seed, split ratios, cost assumptions) + `config.py` loader | AC: unit test loads config
- [x] **T0.5** `Makefile` with `setup`, `lint`, `test` targets; pre-commit with ruff | AC: `make lint test` passes
- [x] **T0.6** Download dataset to `data/raw/telco_churn.csv` (`src/churnguard/data/load.py`) | AC: file present, 7,043 rows

## Phase 1: Data Understanding + EDA (W1)
- [x] **T1.1** `validate.py` with pandera schema from `05_DATA_SPEC.md` section 4 | AC: passes on raw data; tests cover a bad-value case
- [x] **T1.2** Cleaning function (TotalCharges fix, target to 0/1) | AC: no nulls; test covers blank TotalCharges
- [x] **T1.3** `split.py` stratified 70/15/15 saved to `data/processed/` | AC: churn rate within plus or minus 1% across splits
- [x] **T1.4** EDA notebook following `05_DATA_SPEC.md` section 6 | AC: all checklist items done, figures in `reports/figures/`
- [x] **T1.5** Write 5+ business insights | AC: in notebook + progress log
- [x] **T1.6** Malaysia context: load data.gov.my subscribers CSV, plot postpaid vs prepaid trend | AC: chart in `reports/figures/`, used in README

## Phase 2: Baseline (W2)
- [x] **T2.1** `FeatureEngineer` transformer (features in `05_DATA_SPEC.md` section 5) | AC: unit tests per feature
- [x] **T2.2** `ColumnTransformer` preprocessing in `features/build.py` | AC: fit on train only; output shape test
- [x] **T2.3** `evaluate.py` metrics (PR-AUC, ROC-AUC, Brier, Lift@10, Recall@20) | AC: tested on toy arrays
- [x] **T2.4** `train.py` with CV + MLflow logging | AC: E00, E01, E02 logged; results table updated

## Phase 3: Modelling + Tuning (W2-W3)
- [x] **T3.1** Run E03, E04, E05 | AC: logged; results table updated
- [x] **T3.2** Run E06 (SMOTE inside CV via imblearn Pipeline) | AC: logged; decision recorded
- [x] **T3.3** `tune.py` Optuna (E07) | AC: best params saved; MO1 met
- [x] **T3.4** E08 fairness ablation | AC: decision D-006 updated
- [x] **T3.5** Check MO2, MO4 on CV | AC: met or reason logged

## Phase 4: Evaluation, Business and Explainability (W3-W4)
- [x] **T4.1** Calibration on val (E09) | AC: Brier improves; reliability plot saved (MO3)
- [x] **T4.2** `threshold.py` profit curve + optimal threshold + sensitivity | AC: plot saved; threshold in `model_meta.json`
- [x] **T4.3** `shap_explain.py`: global summary + per-customer top 3 reasons with plain-language templates | AC: reasons readable for 5 sample customers
- [x] **T4.4** Final test evaluation with bootstrap CI | AC: `reports/final_metrics.json`; results table filled
- [x] **T4.5** Error analysis + fairness metrics by gender / SeniorCitizen | AC: notebook 02 done; NFR7 checked
- [x] **T4.6** Save final pipeline `models/model.joblib` + `model_meta.json`; register in MLflow | AC: reload and predict test passes

## Phase 5: Serving (W4)
- [x] **T5.1** `predict.py` (load model, predict, tier, reasons) | AC: unit tests
- [x] **T5.2** FastAPI `api/main.py` + `schemas.py` (endpoints per design section 7) | AC: `/docs` works; invalid input returns 422
- [x] **T5.3** Batch CLI `make score FILE=...` | AC: ranked CSV output
- [x] **T5.4** API tests with TestClient | AC: coverage at least 70% overall (EO3)
- [x] **T5.5** Latency check (100 requests) | AC: p95 under 100 ms (EO2)
- [x] **T5.6** Dockerfile (slim, non-root) + `make docker-build docker-run` | AC: container serves `/health`

## Phase 6: CI/CD + Deploy (W5)
- [ ] **T6.1** GitHub Actions: lint + test + coverage gate | AC: green badge in README | **REOPENED 05 Oct (F1, see 13_MODEL_REVIEW.md)**
- [ ] **T6.2** Docker build + smoke test job | AC: passes on main | **REOPENED 05 Oct (F1, see 13_MODEL_REVIEW.md)**
- [ ] **T6.3** Deploy to GCP Cloud Run (manual first, then tag-triggered) | AC: public `/docs` URL works (EO4) | **REOPENED 05 Oct (F1, see 13_MODEL_REVIEW.md)**

## Phase 9: Hardening (EXECUTE NEXT, before reopened Phase 6 and Phase 8 tasks)
> Plan and design: `14_IMPROVEMENT_PLAN.md`. Findings: `13_MODEL_REVIEW.md`. Tick only with evidence in progress log.

### Sprint A: Repo integrity
- [x] **T9.1** Verify Phase 7 commit (0eba781) and commit review docs; working tree clean | AC: `pytest` green; `git status` clean (F8)
- [x] **T9.2** Clean repo: remove duplicate raw files and duplicate figures, Docker `python:3.11-slim`, `requires-python >=3.11`, rebuild image | AC: Docker smoke test passes (F10)

### Sprint B: Model integrity
- [ ] **T9.3** Tie-aware `recall_at_k` / `lift_at_k` + `n_unique_probs` in `evaluate.py` | AC: test proves row-order invariance (F2)
- [ ] **T9.4** E10 calibration redesign: uncalibrated vs sigmoid cv=5 vs isotonic cv=5 on train+val, OOF metrics | AC: rule in plan 4.2 applied; unique probs at least 200; OOF ECE reported; MLflow logged (F2, F3)
- [ ] **T9.5** Re-optimise profit threshold on OOF calibrated probs + sensitivity | AC: `optimal_threshold.json` has `source: oof_train_val` (F7)
- [ ] **T9.6** E11 champion re-decision: E02 vs E07 on identical folds, paired PR-AUC diff | AC: D-013 recorded with rule from plan 4.4; MO1 stated honestly (F5)
- [ ] **T9.7** E12 fairness audit (M0 / M1 / M2) on OOF at chosen threshold | AC: trade-off table; D-014 recorded; gaps at most 0.05 or documented (F6)
- [ ] **T9.8** Fix objective checks in `evaluate_test.py` (same-split, no hardcoded constants) | AC: unit tests; mapping in plan 4.6 (F4)
- [ ] **T9.9** v1.1 final test evaluation (once), champion + runner-up; regenerate model, meta, metrics, scored.csv, figures | AC: D-015 discloses test reuse; API tests pass
- [ ] **T9.10** `docs/MODEL_CARD.md`: intended use, data, metrics with CI, fairness table, limitations | AC: complete
- [ ] **T9.11** Update `06_EXPERIMENT_PLAN.md` results (E10 to E12, v1.1), README results + limitations + "what did not work" | AC: no `__` left in results section (F9)

### Sprint C: Visibility
- [ ] **T9.12** Create GitHub repo, push, CI green on GitHub, badges in README; re-tick T0.1, T6.1, T6.2 | AC: Actions run URL in progress log (F1)
- [ ] **T9.13** GCP project, Artifact Registry, service account secrets, budget alert RM5, deploy; re-tick T6.3 (fallback: Render) | AC: public `/docs` returns 200, `/predict` works (F1)
- [ ] **T9.14** Readiness review: rescore rubric in `13_MODEL_REVIEW.md` section 6 | AC: all gates G1 to G8 pass; score at least 8/10

## Phase 7: Monitoring + Dashboard (W5-W6)
- [x] **T7.1** Simulate drifted batch | AC: script documented
- [x] **T7.2** `drift.py` Evidently report | AC: HTML in `reports/drift/` (EO5)
- [x] **T7.3** Streamlit dashboard (needed for public demo) | AC: upload CSV or form input, ranked list, reasons, segment charts

## Phase 8: Portfolio Packaging (W6)
- [x] **T8.1** README: problem, results table, architecture diagram, how to run, limitations | AC: complete
- [ ] **T8.2** Demo GIF / screenshots of API + dashboard | AC: in README | **REOPENED 05 Oct (F11: no GIF / screenshots yet)**
- [ ] **T8.3** Clean repo, make public, pin repo on GitHub | AC: done | **REOPENED 05 Oct (F11: no git remote; repo not public)**
- [ ] **T8.4** LinkedIn post + add to CV and portfolio spreadsheet | AC: posted; tracker status = Done | **REOPENED 05 Oct (F11: post drafted, not posted)**
- [ ] **T8.5** Deploy Streamlit demo to Hugging Face Spaces | AC: public Space URL loads and scores a sample | **REOPENED 05 Oct (F11: entrypoint only; Space not deployed)**
- [ ] **T8.6** Publish Kaggle notebook (EDA + model summary, link to GitHub) | AC: public notebook URL | **REOPENED 05 Oct (F11: notebook file only; not published on Kaggle)**
- [ ] **T8.7** (Optional) Medium / dev.to write-up | AC: published, linked in README | **REOPENED 05 Oct (F11: article drafted, not published)**
- [ ] **T8.8** Run pre-publish checklist in `10_PUBLISHING_PLAN.md` section 3 | AC: all items ticked | **REOPENED 05 Oct (F11: checklist needs live links)**
