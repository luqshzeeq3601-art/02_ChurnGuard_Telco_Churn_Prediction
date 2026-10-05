# Model Card: ChurnGuard (v1.1.0)

> Structured according to the Model Card standard (Mitchell et al., 2019).

---

## 1. Model Details

- **Model Name**: ChurnGuard Customer Retention Predictor
- **Version**: 1.1.0 (Phase 9 Hardening Release)
- **Model Type**: Binary Classification & Probability Calibration
- **Champion Architecture**: Scikit-Learn `Pipeline` with `FeatureEngineer`, `ColumnTransformer` (StandardScaler + OneHotEncoder), `LogisticRegression(class_weight="balanced")`, wrapped in `CalibratedClassifierCV(method="sigmoid", cv=5)`
- **Runner-Up Architecture**: `LGBMClassifier` (Optuna-tuned 60 trials) wrapped in `CalibratedClassifierCV(method="isotonic", cv=5)`
- **License**: MIT
- **Contact / Maintainer**: NusaTel ML Engineering Team
- **Date**: October 2026

---

## 2. Intended Use

### Primary Intended Uses
- Proactive identification of telco subscribers at high risk of contract termination within the subsequent billing cycle.
- Prioritization of customer retention outreach for customer service and retention marketing teams.
- Optimized allocation of retention campaign budgets using cost-sensitive decision thresholding ($\tau^* = 0.1882$).

### Out-of-Scope & Prohibited Uses
- Credit scoring, loan origination, or underwriting decisions.
- Punitive pricing or adverse service degradation based on predicted churn risk.
- Application of group-specific decision thresholds based on protected demographic attributes (e.g., gender, age).

---

## 3. Training & Evaluation Data

- **Primary Dataset**: IBM Telco Customer Churn dataset framed under fictional Malaysian telco **NusaTel**.
- **External Context**: Malaysian telecom macroeconomic statistics from [data.gov.my](https://data.gov.my).
- **Dataset Size**: 7,043 records (72.6% retained, 27.4% churned).
- **Split Discipline**:
  - `train + val` (5,986 records, 85%): used for 5-fold Stratified cross-validation, hyperparameter tuning, probability calibration, and out-of-fold threshold optimization.
  - Held-out `test` (1,057 records, 15%): touched strictly once for final model evaluation.
- **Fairness Preprocessing**: Demographic attributes `gender` and `SeniorCitizen` are excluded from the Champion model feature set (Fairness Mitigation Option M2) to reduce disparate recall impact.

---

## 4. Performance & Benchmark Metrics

Evaluated on the held-out test set ($N = 1,057$, 281 churners) with **1,000x percentile bootstrap 95% Confidence Intervals**:

| Metric | Champion (Logistic Regression M2) | 95% Bootstrap CI | Runner-up (Tuned LightGBM) | Target / Threshold |
|---|---|---|---|---|
| **ROC-AUC** | **0.8449** | [0.8207, 0.8714] | 0.8473 | $\ge 0.84$ (MO2 Met) |
| **PR-AUC** | **0.6739** | [0.6203, 0.7262] | 0.6760 | $\ge 0.62$ (MO2 Met) |
| **Brier Score** | **0.1361** | [0.1236, 0.1476] | 0.1340 | $\le 0.1542$ (MO3 Met) |
| **Lift @ 10%** | **2.84x** | [2.52x, 3.18x] | 2.98x | $\ge 2.50\text{x}$ (MO2 Met) |
| **Recall @ 20%** | **48.40%** | [44.21%, 53.26%] | 51.60% | $\ge 50.00\%$ |
| **Optimal Threshold ($\tau^*$)** | **0.1882** | — | 0.1882 | Profit-maximising on OOF |
| **Precision @ $\tau^*$** | **47.53%** | [43.42%, 51.91%] | 49.79% | — |
| **Recall @ $\tau^*$** | **88.97%** | [85.00%, 92.44%] | 85.41% | — |
| **F1 Score @ $\tau^*$** | **0.6196** | [0.5799, 0.6576] | 0.6291 | — |
| **Expected Profit / 1k** | **RM36,720.15** | [RM30,063.54, RM42,997.04] | RM37,613.72 | Beats Contact All (RM18,193) |

### Baseline Policy Comparisons (on Held-out Test Split)
- **Optimal Policy ($\tau^* = 0.1882$)**: Expected Profit = **RM36,720.15 / 1,000 customers**
- **Contact All Policy ($\tau = 0.00$)**: Expected Profit = **RM18,193.09 / 1,000 customers**
- **Default Policy ($\tau = 0.50$)**: Expected Profit = **RM29,830.03 / 1,000 customers**
- **Contact None Policy ($\tau = 1.00$)**: Expected Profit = **RM0.00**

---

## 5. Fairness Audit & Demographic Slicing (Experiment E12)

Fairness audit conducted across out-of-fold calibrated predictions at $\tau^* = 0.1882$:

| Option | Features Dropped | Profit / 1k (RM) | Gender Recall Gap | Senior Recall Gap | Max Recall Gap | Status |
|---|---|---|---|---|---|---|
| **M0** | None (All features) | RM38,660.32 | 0.0196 | 0.0893 | 0.0893 | Baseline |
| **M1** | `gender` | RM38,771.27 | 0.0221 | 0.0909 | 0.0909 | Evaluated |
| **M2** | `gender` + `SeniorCitizen` | **RM38,693.49** | **0.0195** | **0.0744** | **0.0744** | **Selected (D-014)** |

### Disparity Analysis & Rationale
- **Gender**: Recall gap is **0.0195** (Male 87.06% vs Female 89.02%), comfortably satisfying the non-functional requirement $\text{gap} \le 0.05$ (NFR7).
- **Senior Citizens**: Recall gap is **0.0744** (Senior 93.58% vs Non-Senior 86.14%). This remaining gap is documented as a known limitation driven by ground-truth base rate disparity: Senior citizens have an actual churn rate of **41.3%** compared to **23.6%** for non-seniors (a 1.75x ratio), primarily concentrated in month-to-month fiber optic contracts lacking technical support bundles.
- **Group-Specific Thresholds**: Explicitly rejected per Decision **D-014** to prevent discriminatory pricing or differential treatment under Malaysian Fair Trade practices.

---

## 6. Model Decisions & Evolution

- **D-013 (Champion Re-Decision)**: In paired 5-fold CV (E11), LightGBM achieved a mean PR-AUC gain over Logistic Regression of only $+0.0045 \pm 0.0078$. Because this gain was less than 1 standard deviation and below the pre-registered $+0.01$ threshold, Logistic Regression was selected as Champion for its superior interpretability, low latency, and zero dependency overhead. LightGBM is maintained as Runner-up.
- **D-014 (Fairness Mitigation)**: Option M2 selected, eliminating `gender` and `SeniorCitizen` from feature inputs while preserving campaign profitability.
- **D-015 (Test Split Disclosure)**: In v1.0, probability calibration suffered from in-sample fit and probability plateau collapse. The test split was re-evaluated once for the hardened v1.1 release following cross-validated calibration redesign (E10).

---

## 7. Known Limitations & Caveats

1. **Synthetic Base Data**: IBM Telco is a semi-synthetic public benchmark; currency and regional variables have been adapted to Malaysian Ringgit (RM) with Malaysian macroeconomic context from data.gov.my.
2. **Fixed Offer Assumptions**: Business metrics assume a standard retention offer cost of RM50 and a customer acceptance rate of 30%. Sensitivity analysis confirms profitability between 20% and 40% acceptance rates.
3. **Data Drift**: Model performance requires ongoing monitoring via Evidently AI for population shift in `MonthlyCharges`, `tenure`, and payment methods.
