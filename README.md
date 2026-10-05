# ChurnGuard: Telco Customer Churn Prediction and Retention Targeting

End-to-end classical ML system that predicts which telco customers will churn, explains why, and ranks them for a cost-aware retention campaign. Served as a REST API in Docker with MLflow tracking, CI/CD and drift monitoring.

## 1. Status
| Item | Value |
|---|---|
| Phase | Phase 0: Setup (see `docs/09_PROGRESS_LOG.md`) |
| Owner | ZeeqRyz |
| Start | Oct 2026 |
| Target finish | 6 weeks from start |

## 2. Highlights (fill in after completion)
- Model: LightGBM tuned with Optuna, ROC-AUC `__`, PR-AUC `__`
- Business: top-20% targeting captures `__`% of churners, expected profit RM `__` per 1,000 customers
- Stack: Python, scikit-learn, LightGBM, SHAP, MLflow, FastAPI, Docker, GitHub Actions, Evidently, GCP Cloud Run, Streamlit (full list: `docs/00_TECH_STACK.md`)

## 3. Live Links (filled in Phase 8)
| Platform | Link |
|---|---|
| Demo (Hugging Face Spaces) | `__` |
| API docs (Cloud Run) | `__` |
| Kaggle notebook | `__` |
| Write-up | `__` |

## 4. Data
- Modelling: IBM Telco Customer Churn (Kaggle), framed as fictional Malaysian telco "NusaTel"
- Malaysia context: data.gov.my Cellular Subscribers by Plan Type (CC BY 4.0)
- No public customer-level Malaysian churn data exists (PDPA 2010)

## 5. Docs
Start with `CLAUDE.md`. All planning lives in `docs/`.

## 6. Quickstart (filled during Phase 5)
```bash
make setup
make train
make serve      # http://localhost:8000/docs
```
