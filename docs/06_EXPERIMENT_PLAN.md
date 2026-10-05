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

### Final Test Results (once)
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
