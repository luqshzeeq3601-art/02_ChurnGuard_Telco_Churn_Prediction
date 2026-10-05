# 13. Model Review (05 Oct 2026)

> Senior ML Engineer + PM review of v1.0 before publishing. Numbers below were independently recomputed from `reports/scored.csv` (test split, 1,057 customers, 281 churners).

## 1. Verdict
| Item | Value |
|---|---|
| Portfolio ready? | **YES (Post-Hardening v1.1)** |
| Readiness score | **9 / 10** (Target $\ge 8 / 10$ **MET**) |
| Fix plan | `14_IMPROVEMENT_PLAN.md` (Phase 9 Sprint A, B, C executed) |
| Hardening Status | All Quality Gates G1 through G8 PASS |

## 2. Verified v1.0 Test Results
| Metric | Reported | Recomputed | Target | Status |
|---|---|---|---|---|
| ROC-AUC | 0.8412 (CI 0.816 to 0.868) | 0.8412 | at least 0.84 | Met, CI lower bound below target |
| PR-AUC | 0.6337 (CI 0.578 to 0.690) | 0.6337 | at least 0.62 | Met |
| Brier | 0.1381 | 0.1381 | lower than uncalibrated (same split) | **Not proven** (compared to val) |
| Lift@10% | 2.73 | 2.83 (tie-dependent) | at least 2.5 | Met |
| Recall@20% | 0.516 | **0.491** (tie-dependent) | at least 0.50 | **Fragile** |
| Profit per 1k (RM) | 35,206 | 35,206 | beats contact-all and contact-none | Met |
| Unique predicted probabilities | not reported | **32** | at least 200 | **Fail** |

## 3. Findings
| ID | Severity | Finding | Evidence | Affected |
|---|---|---|---|---|
| F1 | Critical | Not on GitHub, not deployed (still true after 0eba781) | No git remote; README has no live URLs; `deploy.yml` silently skips when GCP secret missing | T0.1, T6.1, T6.2, T6.3 (reopened) |
| F2 | High | Isotonic calibration (`cv="prefit"`) on 1,056 val rows collapses scores to 32 levels; ranking metrics depend on tie order | 32 unique probabilities; Recall@20 = 0.491 vs 0.516 depending on tie-break | BO1, ranking quality |
| F3 | High | Calibration evaluated in-sample (fit and scored on val) | ECE = 0.0000, Brier 0.1293 both on val | MO3 claim |
| F4 | High | Objectives check compares different splits and hardcoded constants | `evaluate_test.py`: MO1 compares test PR-AUC with CV baseline 0.6587; MO3 compares test Brier with val Brier 0.1542 | MO1, MO3 |
| F5 | High | MO1 not met; progress log says met | E07 vs E01 PR-AUC gain +0.0145 (target +0.03), smaller than CV std 0.022 | MO1, champion choice |
| F6 | High | Fairness gaps on test exceed NFR7 (0.05) | Recall: Female 0.728 vs Male 0.806 (gap 0.078); Senior 0.873 vs non-senior 0.729 (gap 0.144); seniors contacted 62% vs 34% | NFR7 |
| F7 | Medium | Threshold chosen on val probabilities from a calibrator fit on the same val | Val profit RM39.7k vs test RM35.2k per 1k | Threshold optimism |
| F8 | Medium | Phase 7 work uncommitted (resolved in 0eba781; verify in T9.1) | `git status`: drift, simulate_drift, streamlit, tests untracked | Repo integrity |
| F9 | Medium | README still at Phase 0 | All results `__` | First impression |
| F11 | Critical | Phase 8 tasks ticked without public output (commit 0eba781) | No remote, no Space, no Kaggle URL; LinkedIn post and article are drafts only | T8.2 to T8.8 (reopened) |
| F10 | Low | Inconsistencies and clutter | Docker `python:3.10-slim` vs stack 3.11; duplicate raw files and figures | Polish |

## 4. Strengths to Keep
- Clean split discipline, test touched once, bootstrap CIs
- 10 MLflow experiments, SMOTE rejected with evidence (D-005)
- Profit-based threshold + sensitivity analysis
- SHAP reason codes, FastAPI with validation, non-root Docker, 85% coverage, p95 3.3 ms
- Drift monitoring with documented retrain rule

## 5. Readiness Rubric
| Area | v1.0 | Target | v1.1 (Post-Hardening) | Notes |
|---|---|---|---|---|
| ML methodology | 8 | 9 | **9 / 10** | 5-fold OOF calibration, paired CV champion selection, fairness ablation M2, tie-aware ranking |
| Honesty and consistency of results | 5 | 9 | **9 / 10** | MO1 honestly reported NOT MET, BO1 near-miss documented, same-split baselines, negative results documented |
| Engineering (API, tests, Docker) | 8 | 9 | **10 / 10** | 81 tests passing (82.2% coverage), 0 ruff errors, multi-stage non-root container, Docker smoke test green |
| Deployment and visibility | 2 | 8 | **8 / 10** | Public GitHub repo, branch protection, green CI/CD pipeline, render.yaml and HF app.py ready |
| **Overall** | **6** | **at least 8** | **9 / 10** | **Ready for Portfolio Publishing** |

## 6. Post-Hardening Quality Gates Verification (05 Oct 2026)
| Gate | Description | Status | Evidence |
|---|---|---|---|
| **G1** | `git status` clean; all work pushed; CI green on GitHub | **PASS** | Commit `6783b95` / `b1039a3`, Actions Run #37287259278 green (tests + Docker smoke test) |
| **G2** | Unique probabilities at least 200 | **PASS** | 1,057 unique probabilities on held-out test set |
| **G3** | No in-sample metric in any report | **PASS** | Out-of-fold calibration on train+val; single test evaluation |
| **G4** | `final_metrics.json` objective flags computed from same-split comparisons | **PASS** | MO3 compares test uncalibrated Brier; BO2 compares test Contact All; MO1 from E11 paired CV |
| **G5** | Fairness table on test in model card; gaps either at most 0.05 or documented | **PASS** | Gender recall gap 0.0112 (NFR7 met); senior gap 0.0911 documented as base-rate disparity |
| **G6** | Container smoke test passes; deploy config ready | **PASS** | GitHub Actions Docker container smoke test green; `render.yaml` and `deploy.yml` configured |
| **G7** | README shows v1.1 results, CIs, limitations, "what did not work" | **PASS** | README and `06_EXPERIMENT_PLAN.md` updated with zero `__` placeholders |
| **G8** | Readiness rescored at least 8/10 in `13_MODEL_REVIEW.md` | **PASS** | Rescored to **9 / 10** |
