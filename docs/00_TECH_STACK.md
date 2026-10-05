# 00. Technology Stack

> Single source of truth for tools. Pin exact versions in `requirements.txt` at task T0.2 (latest stable at that date), then record them in the "Pinned" column.

## 1. Core Stack
| Layer | Tool | Used for | Phase | Pinned |
|---|---|---|---|---|
| Language | Python 3.10 / 3.11 | Everything | All | 3.10.11 |
| Environment | venv + pip | Isolated deps | P0 | pip 24.x |
| Version control | Git + GitHub | Code, PRs, CI | All | Git 2.x |
| Data | pandas, numpy | Data handling | P1+ | 2.2.3, 1.26.4 |
| Validation | pandera | Schema checks | P1 | 0.21.0 |
| Visualisation | matplotlib, seaborn, plotly | EDA, reports | P1, P4 | 3.9.3, 0.13.2, 5.24.1 |
| ML | scikit-learn | Pipelines, baselines, metrics, calibration | P2+ | 1.5.2 |
| ML | LightGBM, XGBoost | Gradient boosting models | P3 | 4.5.0, 2.1.3 |
| Imbalance | imbalanced-learn | SMOTE experiment (E06) | P3 | 0.12.4 |
| Tuning | Optuna | Hyperparameter search | P3 | 4.1.0 |
| Explainability | SHAP | Global + per-customer reasons | P4 | 0.46.0 |
| Experiment tracking | MLflow (local file store) | Runs, metrics, model registry | P2+ | 2.18.0 |
| Config | PyYAML | `configs/config.yaml` | P0+ | 6.0.2 |
| API | FastAPI, Pydantic, Uvicorn | REST serving | P5 | 0.115.5, 2.10.2, 0.32.1 |
| Dashboard | Streamlit | Public demo UI | P7 | 1.40.2 |
| Monitoring | Evidently | Drift reports | P7 | 0.4.40 |
| Testing | pytest, pytest-cov, httpx | Unit + API tests | P0+ | 8.3.4, 6.0.0, 0.28.0 |
| Code quality | ruff, pre-commit | Lint + format | P0+ | 0.8.3, 4.0.1 |
| Automation | Make (GNU Make; on Windows use Git Bash or `make` via Chocolatey) | Task runner | P0+ | GNU Make |
| Container | Docker (Docker Desktop on Windows) | Packaging | P5 | Docker 24+ |
| CI/CD | GitHub Actions | Lint, test, build, deploy | P6 | Ubuntu latest |
| Cloud (API) | GCP Cloud Run + Artifact Registry | Public REST API | P6 | GCP serverless |
| Hosting (demo) | Hugging Face Spaces | Public Streamlit demo | P8 | HF Spaces |
| Notebooks | Jupyter / VS Code | EDA, error analysis | P1, P4 | VS Code |

## 2. Dev Environment (Windows)
| Item | Choice |
|---|---|
| OS | Windows 10/11 |
| IDE | VS Code (Python, Jupyter, Docker extensions) |
| Shell | Git Bash or WSL2 (for `make`) |
| Python install | python.org 3.11 or `pyenv-win` |

## 3. Why This Stack (job-market mapping)
| Tool | Seen in Malaysian postings |
|---|---|
| scikit-learn, XGBoost, LightGBM | DKSH, Keysight, Astro, PetBacker |
| MLflow | BSI, DKSH |
| Docker, CI/CD | AirAsia, BSI, DKSH |
| GCP | AirAsia (BigQuery, Vertex AI), PetBacker |
| Drift monitoring | AirAsia, Keysight, BSI |
| Power BI-style communication | DKSH, Astro (covered by Streamlit dashboard) |

## 4. Not Used (deliberately)
| Tool | Reason |
|---|---|
| TensorFlow / PyTorch | Classical ML project; DL is out of scope |
| Kubernetes | Overkill for single service; Cloud Run is enough |
| Airflow / Prefect | Listed as Future Work |
