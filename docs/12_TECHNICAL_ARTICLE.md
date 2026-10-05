# 12. Technical Article: Profit-Driven Churn Modelling for Telecom

> Medium / dev.to publication draft.

# Stop Using 0.5 Thresholds: Building a Profit-Driven Churn Retention System with LightGBM, SHAP, and FastAPI

When data scientists build churn prediction models, the workflow often stops at a high ROC-AUC score and an arbitrary classification threshold of `0.5`.

In real-world telecom operations, this approach fails because **the cost of a False Negative (losing an RM780 customer) is 15 times greater than a False Positive (offering an unnecessary RM50 retention voucher)**.

In this deep dive, we walk through building **ChurnGuard**, an end-to-end ML platform designed to maximize net campaign profit, explain risk drivers to frontline agents, and serve low-latency inferences in production.

---

## 1. Problem Framing & Malaysian Market Dynamics
According to national statistics from [data.gov.my](https://data.gov.my), Malaysia has surpassed 50 million cellular subscriptions (>140% mobile penetration). When subscriber growth plateaus, revenue growth becomes a zero-sum game of customer retention and postpaid ARPU preservation.

We model this scenario using a cohort of 7,043 telecom subscribers under the fictional brand **NusaTel**.

---

## 2. Exploratory Data Analysis & Domain Feature Engineering
Analyzing empirical churn patterns revealed three critical drivers:
1. **Contract Lock-in:** Month-to-month contracts exhibit a **42.9% churn rate** (driving 88.6% of all churners), compared to 10.9% for 1-year and 3.0% for 2-year contracts.
2. **Fiber Optic Service Deficit:** Fiber optic users churn at **41.8%** (vs 19.2% DSL); this spikes to **>48%** when Tech Support is missing.
3. **Payment Friction:** Electronic Check payment method has a **45.6% churn rate** vs ~15% for automated card/bank billing.

We created 8 domain features:
- `tenure_bucket`: Segmenting customer lifecycle (0-6m, 7-12m, 13-24m, 25-48m, 49-72m).
- `charge_increase_ratio`: Monthly spend relative to historical average.
- `fiber_no_support`: Identifying high-spend customers on fiber without technical support.
- `has_protection_bundle`: OnlineSecurity + TechSupport flag.
- `is_auto_pay`: Automated vs manual payment friction indicator.

---

## 3. Modelling, Bayesian Tuning & Probability Calibration
We evaluated candidate algorithms under 5-fold Stratified Cross-Validation:
- **Baseline Dummy:** PR-AUC 0.2690
- **Logistic Regression (Engineered):** PR-AUC 0.6611, ROC-AUC 0.8458
- **Random Forest:** PR-AUC 0.6629, ROC-AUC 0.8462
- **XGBoost:** PR-AUC 0.6631, ROC-AUC 0.8437
- **LightGBM Champion (Optuna Tuned):** PR-AUC **0.6732**, ROC-AUC **0.8468**, Lift@10% **2.92x**.

### Probability Calibration
Tree-based classifiers often produce uncalibrated probabilities. Applying **Isotonic Calibration** reduced the Brier Score from 0.1542 to 0.1293 (16% relative improvement), ensuring predicted probabilities match true empirical event frequencies.

---

## 4. Profit Curve Optimization ($\tau^* = 0.18$)
We define the net campaign profit equation:
$$\text{Profit} = \text{TP} \times (\text{CLV} - \text{Cost}) - \text{FP} \times \text{Cost}$$

Where:
- Retention Offer Cost: RM 50
- Retention Conversion Rate: 30%
- Expected CLV: RM 780

Evaluating across thresholds $\tau \in [0.01, 0.99]$ showed that the profit-maximizing cutoff is **$\tau^* = 0.18$**.
- Standard $\tau = 0.50$: RM 24,420 expected profit per 1,000 customers.
- Optimal $\tau^* = 0.18$: **RM 35,206 expected profit per 1,000 customers (+44.2% uplift)**.

---

## 5. Frontline Explainability with SHAP
Black-box scores do not help call center agents save customers. We paired TreeSHAP with plain-language reason templates to generate the top 3 drivers per customer:
- *"Month-to-month contract increases cancellation risk."*
- *"High monthly charges (RM 89.50) without protective support bundle."*
- *"Tenure under 6 months in critical onboarding window."*

---

## 6. Serving Architecture & Latency
- **FastAPI REST Service:** Endpoints for `/health`, `/model-info`, `/predict`, and `/predict/batch`.
- **Latency:** Benchmark across 100 requests yielded **mean 2.44 ms** and **p95 3.27 ms** (< 100 ms target).
- **Docker & CI/CD:** Slim non-root container, GitHub Actions with 85% test coverage gate.
- **Evidently AI:** Real-time data drift surveillance against baseline training distributions.

---

## 7. Conclusion
Building effective ML products requires connecting algorithmic objectives to business value. By moving from default classification thresholds to calibrated profit curves, ML engineers can directly demonstrate bottom-line ROI to executive stakeholders.

Source Code: https://github.com/ZeeqRyz/02_ChurnGuard_Telco_Churn_Prediction
