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
| E00 | | | |
| E01 | | | |
| E02 | | | |
| E03 | | | |
| E04 | | | |
| E05 | | | |
| E06 | | | |
| E07 | | | |
| E08 | | | |
| E09 | | | |

### Final Test Results (once)
| Metric | Value | 95% CI |
|---|---|---|
| ROC-AUC | | |
| PR-AUC | | |
| Brier | | |
| Lift@10% | | |
| Recall@20% | | |
| Profit-optimal threshold | | |
| Expected profit per 1,000 customers (RM) | | |
