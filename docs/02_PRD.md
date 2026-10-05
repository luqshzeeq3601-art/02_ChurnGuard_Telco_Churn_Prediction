# 02. Product Requirements Document (PRD)

| Field | Value |
|---|---|
| Product | ChurnGuard |
| Version | 1.0 |
| Owner / PM | ZeeqRyz |
| Status | Approved for build |
| Last updated | 05 Oct 2026 |

## 1. Summary
ChurnGuard scores every customer with a churn probability, explains the top reasons, and outputs a ranked, budget-aware contact list for the retention team. Delivered as a REST API, a batch scoring script, and a simple dashboard.

## 2. Users and Personas
| Persona | Need | How ChurnGuard helps |
|---|---|---|
| Retention Manager | Who to call this month within budget | Ranked list + recommended cut-off |
| Customer Service Agent | What to say to this customer | Top 3 reason codes per customer |
| Marketing Analyst | Which segments are leaking | Segment churn insights |
| ML Engineer (maintainer) | Retrain, monitor, deploy safely | MLflow, CI/CD, drift report |

## 3. User Stories
| ID | As a... | I want... | So that... | Priority |
|---|---|---|---|---|
| US1 | Retention Manager | a churn score for each customer | I can focus on high-risk ones | Must |
| US2 | Retention Manager | a recommended number of customers to contact | I maximise profit within budget | Must |
| US3 | Agent | the top reasons a customer is at risk | I can tailor the offer | Must |
| US4 | Analyst | churn rates by segment | I can design targeted plans | Should |
| US5 | Engineer | to score a single customer via API | other systems can integrate | Must |
| US6 | Engineer | to score a CSV in batch | monthly campaigns are easy | Must |
| US7 | Engineer | a drift report on new data | I know when to retrain | Should |
| US8 | Manager | a dashboard to explore scores | I do not need to read code | Must |

## 4. Functional Requirements
| ID | Requirement | Acceptance criteria |
|---|---|---|
| FR1 | Data ingestion and validation | Loads raw CSV, enforces schema in `05_DATA_SPEC.md`, fails loudly on violations |
| FR2 | Feature pipeline | Single sklearn `Pipeline` used in both training and serving; no train/serve skew |
| FR3 | Model training | `make train` trains, tunes, calibrates, logs to MLflow, saves model artifact |
| FR4 | Evaluation report | Outputs ROC-AUC, PR-AUC, Brier, Lift@10%, Recall@20%, confusion matrix at chosen threshold, profit curve |
| FR5 | Threshold optimisation | Selects threshold maximising expected profit using cost assumptions in `06_EXPERIMENT_PLAN.md` |
| FR6 | Explainability | Global SHAP summary + per-customer top 3 reasons in plain text |
| FR7 | Single prediction API | `POST /predict` returns probability, risk tier, top reasons, model version |
| FR8 | Batch prediction | `POST /predict/batch` and CLI `make score FILE=...` return ranked CSV |
| FR9 | Health and metadata | `GET /health`, `GET /model-info` |
| FR10 | Drift monitoring | Script compares reference vs new batch, produces Evidently HTML report |
| FR11 | Dashboard (Must, public demo) | Streamlit app: upload CSV or form input, see ranked list, reasons, segment charts; deployed on Hugging Face Spaces |
| FR12 | Malaysia context | README and dashboard show real Malaysian postpaid vs prepaid trend (data.gov.my) |

### Risk Tiers
| Tier | Rule |
|---|---|
| High | probability at or above profit-optimal threshold |
| Medium | between 0.5 x threshold and threshold |
| Low | below 0.5 x threshold |

## 5. Non-Functional Requirements
| ID | Requirement | Target |
|---|---|---|
| NFR1 | Latency | `/predict` p95 under 100 ms (local) |
| NFR2 | Reproducibility | Seeded; metrics reproducible within plus or minus 0.005 |
| NFR3 | Code quality | ruff clean, type hints, at least 70% test coverage |
| NFR4 | Portability | Runs via `docker run` with no extra setup |
| NFR5 | Security | No secrets in repo; input validated with Pydantic |
| NFR6 | Documentation | README with results, architecture diagram, how to run |
| NFR7 | Fairness check | Report metrics by `gender` and `SeniorCitizen`; flag gaps over 0.05 in recall |

## 6. API Contract (v1)
`POST /predict`
```json
{
  "customerID": "7590-VHVEG",
  "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
  "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
  "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
  "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
  "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85
}
```
Response
```json
{
  "customerID": "7590-VHVEG",
  "churn_probability": 0.71,
  "risk_tier": "High",
  "top_reasons": [
    "Month-to-month contract increases risk",
    "Very short tenure (1 month)",
    "Pays by electronic check"
  ],
  "model_version": "lgbm-v1.0.0"
}
```

## 7. Success Metrics
See `01_PROBLEM_AND_OBJECTIVES.md` section 5 (BO, MO, EO targets).

## 8. Assumptions and Constraints
- No public customer-level Malaysian churn data exists (PDPA 2010); IBM Telco dataset represents a fictional Malaysian telco ("NusaTel"); currency treated as RM for framing. Real Malaysian market data (data.gov.my) used for context only (D-010)
- Tech stack: `00_TECH_STACK.md`; publishing targets: `10_PUBLISHING_PLAN.md`
- Single snapshot dataset, no time column, so "next cycle" churn is approximated by the label
- Solo developer, about 10 to 12 hours per week
- Free-tier cloud only

## 9. Risks
Tracked in `03_PROJECT_PLAN.md` section 6.

## 10. Future Work (out of v1 scope)
- Uplift modelling (who responds to offers, not just who churns)
- Survival analysis (when will they churn)
- Automated retraining pipeline (Airflow / Prefect)
- Feature store, A/B test of real campaign
