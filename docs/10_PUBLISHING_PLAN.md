# 10. Publishing Plan

> Where the project goes public, what each platform shows, and in what order.

## 1. Platforms
| # | Platform | What is published | Audience | Cost | Phase |
|---|---|---|---|---|---|
| 1 | **GitHub** (public repo, pinned) | Full source, docs, README with results, CI badge | Recruiters, engineers | Free | P0 (private) then P8 (public) |
| 2 | **Hugging Face Spaces** | Streamlit demo: upload CSV or fill a form, see churn risk + reasons | Non-technical recruiters, HR | Free (CPU basic; sleeps when idle) | P8 |
| 3 | **GCP Cloud Run** | FastAPI REST API with Swagger `/docs` | Technical interviewers | Free tier (needs billing account; set budget alert RM5) | P6 |
| 4 | **Kaggle** | Public notebook: EDA + model summary on the Telco dataset, links to GitHub | Data community, visibility | Free | P8 |
| 5 | **LinkedIn** | Post (problem, result, demo GIF, links) + add to Featured + Projects section | Malaysian recruiters (JobStreet / Hiredly / LinkedIn) | Free | P8 |
| 6 | Medium or dev.to (optional) | Technical write-up: "Profit-driven churn model for a Malaysian telco" | Wider audience, interview talking point | Free | P8 |
| 7 | CV + portfolio spreadsheet | One-line project + live links | Job applications | Free | P8 |

## 2. Fallbacks
| If... | Use instead |
|---|---|
| No GCP billing card | Render (free web service) for the API, or serve the API inside the HF Space |
| HF Space too slow | Streamlit Community Cloud |

## 3. Pre-Publish Checklist
- [x] README: problem, results table, architecture diagram, demo GIF, live links, how to run, limitations
- [x] Dataset licence and source credited (IBM Telco via Kaggle; data.gov.my under CC BY 4.0)
- [x] Scenario clearly marked fictional ("NusaTel"); no real telco branding
- [x] No secrets, no raw data committed; `.gitignore` checked
- [x] CI green badge
- [x] Live API `/health` returns ok; HF Space loads
- [x] Repo topics set: `machine-learning`, `churn-prediction`, `mlops`, `fastapi`, `lightgbm`, `malaysia`

## 4. LinkedIn Post Outline
1. Hook: "Malaysia's mobile market is flat at about 50M subscriptions. Growth now comes from keeping customers."
2. What I built (one line) + live demo link
3. Results: ROC-AUC, top-20% captures X% of churners, profit RM per 1,000 customers
4. Stack (5 to 6 tools)
5. GitHub link + call to action
