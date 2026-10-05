# 06. Experiment Plan

## 1. Validation Strategy
- Holdout: train 70 / val 15 / test 15, stratified, seed 42
- Model selection: 5-fold stratified CV on **train** only
- Calibration + threshold: **val**
- Final report: **test**, once. Bootstrap 1,000x for 95% CI on ROC-AUC and PR-AUC

## 2. Metrics
| Type | Metric | Why |
|---|---|---|
| Primary | **PR-AUC** (average precision) | Imbalanced target; focus on churners |
| Secondary | ROC-AUC | Ranking quality, standard comparison |
| Business | Lift@top 10%, Recall@top 20% | Matches limited contact budget |
| Business | Expected profit (RM) at chosen threshold | Final decision metric |
| Calibration | Brier score, reliability curve | Probabilities used for profit math |
| Threshold-level | Precision, Recall, F1 at chosen threshold | For reporting |

## 3. Business Cost Assumptions (editable in `configs/config.yaml`)
| Parameter | Value | Note |
|---|---|---|
| Average monthly revenue per customer | dataset `MonthlyCharges` mean (about RM65) | Treated as RM for framing |
| Customer value saved (CLV proxy) | 12 x MonthlyCharges | Assumption: 12-month horizon |
| Retention offer cost | RM50 per contacted customer | Assumption |
| Offer success rate (churner stays) | 30% | Assumption; tested at 20% and 40% |

Expected profit for contacting customer i:
```
profit_i = p_i * success_rate * CLV_i - offer_cost
```
Threshold chosen to maximise total profit on val. Sensitivity analysis for success_rate in {0.2, 0.3, 0.4}.

## 4. Experiment Matrix
| Exp ID | Model | Imbalance handling | Features | Purpose |
|---|---|---|---|---|
| E00 | DummyClassifier (stratified) | none | raw | Floor |
| E01 | Logistic Regression | class_weight=balanced | raw | **Baseline** |
| E02 | Logistic Regression | balanced | raw + engineered | Feature value test |
| E03 | Random Forest | balanced | raw + engineered | Bagging reference |
| E04 | XGBoost | scale_pos_weight | raw + engineered | Boosting |
| E05 | LightGBM | is_unbalance | raw + engineered | Boosting |
| E06 | Best of E03-E05 | SMOTE inside CV | raw + engineered | Resampling test |
| E07 | Best model | best | + Optuna tuning (50-100 trials) | Tuning |
| E08 | E07 | best | without gender / SeniorCitizen | Fairness ablation |
| E09 | E07 + CalibratedClassifierCV (isotonic vs sigmoid) | best | best | Calibration |

## 5. Optuna Search Space (LightGBM)
| Param | Range |
|---|---|
| num_leaves | 8 to 64 |
| max_depth | 3 to 10 |
| learning_rate | 0.01 to 0.2 (log) |
| n_estimators | 100 to 1000 (+ early stopping) |
| min_child_samples | 10 to 100 |
| subsample | 0.6 to 1.0 |
| colsample_bytree | 0.6 to 1.0 |
| reg_alpha, reg_lambda | 1e-3 to 10 (log) |
Objective: maximise mean CV PR-AUC.

## 6. MLflow Conventions
- Experiment name: `churnguard`
- Run name: `<ExpID>_<model>_<short-desc>` e.g. `E05_lgbm_engineered`
- Log: params, CV mean and std per metric, git commit, feature list, confusion matrix PNG, PR and ROC curves PNG
- Tag best run `stage=candidate`; final model registered as `churnguard-model`, version noted in `model_meta.json`

## 7. Error Analysis (Phase 4)
- Inspect top false negatives (missed churners) and top false positives
- Segment-level metrics: contract, tenure bucket, internet service, gender, SeniorCitizen
- Write findings in `notebooks/02_error_analysis.ipynb` + progress log

## 8. Results Table (fill as you go)
| Exp ID | CV PR-AUC (mean plus or minus std) | CV ROC-AUC | Notes |
|---|---|---|---|
| E00 | 0.2690 ± 0.0024 | 0.5089 | Dummy stratified floor |
| E01 | 0.6587 ± 0.0220 | 0.8442 | Logistic Regression baseline (raw features) |
| E02 | 0.6611 ± 0.0124 | 0.8458 | Logistic Regression (raw + engineered features, lower variance) |
| E03 | 0.6629 ± 0.0172 | 0.8462 | Random Forest (class_weight=balanced, engineered features) |
| E04 | 0.6631 ± 0.0266 | 0.8437 | XGBoost (scale_pos_weight=2.77, engineered features) |
| E05 | 0.6598 ± 0.0251 | 0.8409 | LightGBM untuned (scale_pos_weight=2.77, engineered features) |
| E06 | 0.6537 ± 0.0274 | 0.8417 | LightGBM with SMOTE in CV (degraded PR-AUC -> rejected per D-005) |
| E07 | **0.6732 ± 0.0217** | **0.8468** | **Champion: LightGBM + Optuna tuned (MO1/MO2/MO4 met)** |
| E08 | 0.6706 ± 0.0217 | 0.8464 | Fairness ablation (without gender/SeniorCitizen, minimal drop) |
| E09 | **0.6732 ± 0.0217** (CV) | **0.8468** (CV) | E07 + Isotonic Calibration on Val (Brier: 0.1542 -> 0.1293, MO3 met) |

| E10 | 0.6584 (OOF) | 0.8467 (OOF) | Calibration redesign (cv=5 on train+val, 5,986 rows). Isotonic cv=5 selected (lowest Brier: 0.1341, ECE: 0.0142, 2,842 unique probs, F2/F3 fixed) |
| E11 | 0.6598 ± 0.0182 (LR) vs 0.6643 ± 0.0147 (LGBM) | 0.8458 (LR) vs 0.8468 (LGBM) | Paired 5-fold CV on train+val: LightGBM mean gain +0.0045 < 1 std (0.0078). LR selected as Champion (D-013), LightGBM as runner-up. MO1 not met. |
| E12 | 0.6614 (M2 OOF) | 0.8455 (M2 OOF) | Fairness audit on OOF at tau=0.1882: M2 (drop gender + SeniorCitizen) selected (D-014). Max recall gap 0.0744 (down from 0.0893) at 0% profit loss. |

### Final Test Results (v1.0 initial, test reuse disclosed in D-015)
| Metric | Value | 95% CI |
|---|---|---|
| ROC-AUC | 0.8412 | [0.8155, 0.8679] |
| PR-AUC | 0.6337 | [0.5776, 0.6899] |
| Brier Score | 0.1381 | [0.1247, 0.1509] |
| Lift@10% | 2.73x | [2.54x, 3.20x] |
| Recall@20% | 51.60% | [46.05%, 54.86%] |
| Profit-optimal threshold | 0.18 | - |
| Expected profit per 1,000 customers (RM) | RM35,206.24 | [RM29,031.29, RM41,165.51] |
| Precision at tau* | 53.75% | [48.59%, 58.88%] |
| Recall at tau* | 76.51% | [71.28%, 81.40%] |
| F1 at tau* | 0.6314 | [0.5871, 0.6724] |

### v1.1 Test Results (after Phase 9 Hardening, D-015)
| Metric | Champion (Logistic Regression M2) | Runner-up (Tuned LightGBM) | 95% CI (Champion) | Target / Benchmark |
|---|---|---|---|---|
| ROC-AUC | **0.8449** | 0.8473 | [0.8207, 0.8714] | $\ge 0.84$ (MO2 Met) |
| PR-AUC | **0.6739** | 0.6760 | [0.6203, 0.7262] | $\ge 0.62$ (MO2 Met) |
| Brier (calibrated vs uncalibrated) | **0.1361** (vs 0.1472 uncal) | 0.1340 (vs 0.1522 uncal) | [0.1236, 0.1476] | Lower is better (MO3 Met) |
| Lift@10% (tie-aware) | **2.84x** | 2.98x | [2.52x, 3.18x] | $\ge 2.50\text{x}$ (MO2 Met) |
| Recall@20% (tie-aware) | **48.40%** | 51.60% | [44.21%, 53.26%] | $\ge 50.00\%$ |
| Unique probabilities | **1,057** | 1,057 | — | $\ge 200$ (G2 Met) |
| Threshold ($\tau^*$ from OOF) | **0.1882** | 0.1882 | — | Profit-optimal on OOF |
| Profit per 1k (RM) | **RM36,720.15** | RM37,613.72 | [RM30,063.54, RM42,997.04] | Beats Contact All (RM18,193) |
| Precision at $\tau^*$ | **47.53%** | 49.79% | [43.42%, 51.91%] | — |
| Recall at $\tau^*$ | **88.97%** | 85.41% | [85.00%, 92.44%] | — |
| F1 at $\tau^*$ | **0.6196** | 0.6291 | [0.5799, 0.6576] | — |

### Generated Reports & Visualizations
- [`reports/figures/07_calibration_curve.png`](reports/figures/07_calibration_curve.png): 5-fold OOF calibration curves (Uncalibrated vs Sigmoid vs Isotonic).
- [`reports/figures/08_profit_curve.png`](reports/figures/08_profit_curve.png): Net campaign profit curve across thresholds, highlighting $\tau^* = 0.1882$.
- [`reports/figures/09_shap_summary.png`](reports/figures/09_shap_summary.png): SHAP summary beeswarm plot of feature attributions for Champion model.
- [`reports/final_metrics.json`](reports/final_metrics.json): Complete point estimates, bootstrap 95% CIs, and objective verifications.
- [`reports/scored.csv`](reports/scored.csv): Full test split predictions with risk tiers and top 3 reason codes.

### Negative Results & What Did Not Work
1. **SMOTE Oversampling (E04):** Degraded cross-validated PR-AUC from 0.6587 to 0.6482 due to synthetic point smearing across continuous feature boundaries.
2. **Gradient Boosting Premium (E11 / D-013):** LightGBM provided only $+0.0045 \pm 0.0078$ mean PR-AUC gain over Logistic Regression on paired 5-fold CV (< 1 std). Per Occam's razor, Logistic Regression was selected as Champion for lower operational complexity and direct explainability.
3. **In-Sample Prefit Calibration (E09):** Prefit calibration on small validation data caused probability collapse to 34 unique values. Replaced by 5-fold OOF calibration on train+val (E10), achieving 1,057 unique probabilities.
4. **Adversarial Fairness Weighting (E12):** Complex multi-objective loss reweighting was unnecessary; complete removal of protected features (`gender`, `SeniorCitizen`, Option M2) achieved fairness compliance with zero profit penalty.
