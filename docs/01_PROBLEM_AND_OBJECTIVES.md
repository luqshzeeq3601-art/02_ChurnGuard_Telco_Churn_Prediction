# 01. Problem and Objectives

## 1. Business Context
- Malaysian mobile market is saturated: total subscribers flat at about **50 million** (2023 to 2025)
- Growth is shifting to **postpaid**: postpaid base grew from 14.8M to 16.9M (29.6% to 33.8% share), while prepaid fell by more than 2.4M
- In a flat market, growth comes from **keeping and upgrading existing customers**, not from new users. Every postpaid customer lost is recurring revenue lost
- Acquiring a new customer typically costs several times more than retaining one (industry rule of thumb, treat as assumption)

Source: The Star, "Postpaid services key growth driver for mobile network operators", 15 Jul 2026.

## 2. Scenario (fictional company, used for framing)
**"NusaTel"**, a fictional Malaysian telco with about 7,000 postpaid / contract customers in this sample.
- Retention team can contact only a limited number of customers per month
- Current approach: blanket discounts or random calls, so budget is wasted on customers who would stay anyway
- No visibility into **why** customers leave

## 3. Problem Statement
> NusaTel loses roughly 1 in 4 customers, but cannot tell **who** is about to leave or **why**. Retention budget is spent without targeting, so most offers go to customers who would have stayed anyway, and many at-risk customers are never contacted.

## 4. Root Questions the Project Answers
| # | Question | Output |
|---|---|---|
| Q1 | Who is likely to churn in the next cycle? | Churn probability per customer |
| Q2 | Why is this customer at risk? | Top 3 SHAP reason codes per customer |
| Q3 | Whom should we contact, given limited budget? | Ranked list + profit-optimal threshold |
| Q4 | Which segments churn most? | Segment insights (contract, tenure, payment, services) |
| Q5 | Is the model still valid over time? | Drift report |

## 5. Objectives (SMART)
### 5.1 Business Objectives
| ID | Objective | Target |
|---|---|---|
| BO1 | Concentrate churners in a small contact list | Top 20% of ranked customers capture **at least 50%** of churners |
| BO2 | Maximise retention campaign profit | Profit-optimal threshold beats "contact everyone" and "contact no one" by a clear margin (RM, per 1,000 customers) |
| BO3 | Explain churn drivers to non-technical staff | Top 5 global drivers + per-customer reasons in plain language |

### 5.2 ML Objectives
| ID | Objective | Target |
|---|---|---|
| MO1 | Beat baseline | Tuned model PR-AUC at least **+0.03** over logistic regression baseline |
| MO2 | Ranking quality | ROC-AUC **at least 0.84**, PR-AUC **at least 0.62**, Lift@top-decile **at least 2.5** on held-out test |
| MO3 | Probability quality | Calibrated: Brier score lower than uncalibrated model |
| MO4 | Robustness | 5-fold CV std of PR-AUC **at most 0.03** |

### 5.3 Engineering Objectives
| ID | Objective | Target |
|---|---|---|
| EO1 | Reproducible training | One command (`make train`) reproduces metrics within plus or minus 0.005 |
| EO2 | Serving | FastAPI `/predict` p95 latency **under 100 ms** locally |
| EO3 | Quality | Test coverage **at least 70%** on `src/`; CI green on every push |
| EO4 | Deployment | Container live on GCP Cloud Run (free tier) with public `/docs` |
| EO5 | Monitoring | Evidently drift report generated from a simulated new batch |

## 6. Success Definition
Project is **done** when all MO and EO targets are met (or a documented reason exists in `08_DECISIONS_LOG.md`), README shows results, and a recruiter can call the live API.

## 7. Out of Scope
- Real customer data from any real telco
- Uplift / causal modelling (listed as Future Work)
- Deep learning models
- Real-time streaming pipelines

## 8. Skills Demonstrated (portfolio mapping)
| Job requirement (Malaysia postings) | Shown by |
|---|---|
| Churn / propensity modelling (AirAsia, Astro) | Core model |
| scikit-learn, XGBoost, LightGBM (DKSH, Keysight) | Model comparison |
| A/B-style statistical thinking (AirAsia) | Threshold and profit analysis, CI on metrics |
| MLflow, CI/CD, Docker (BSI, AirAsia, DKSH) | MLOps layer |
| Drift monitoring (AirAsia, Keysight) | Evidently report |
| Cloud deploy (PetBacker, AirAsia) | Cloud Run |
