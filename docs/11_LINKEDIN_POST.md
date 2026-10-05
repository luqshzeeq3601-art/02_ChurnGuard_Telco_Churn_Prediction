# 11. LinkedIn Post & Portfolio Publication Draft

> Ready to copy-paste for LinkedIn publication, portfolio update, and resume bullet points.

---

### LinkedIn Post Draft

```markdown
Malaysia's mobile market is flat at ~50M subscriptions and >140% penetration (data.gov.my). Net subscriber growth has peaked—so in telecom today, real revenue growth comes from keeping high-value postpaid customers.

I built **ChurnGuard** 🛡️—a production-grade, cost-aware churn prediction and retention targeting platform for a Malaysian telco ("NusaTel").

Here is what makes this different from standard churn projects:

1. 💰 **Profit-Optimized Decisions over Default 0.5 Cutoffs**:
   Lost customers (RM780 CLV) cost 15x more than a proactive retention offer (RM50). By optimizing over the expected campaign profit curve, our optimal cutoff ($\tau^* = 0.18$) delivers **RM 35,206 net profit per 1,000 customers** (+44% gain over standard 0.5 threshold).

2. 🎯 **High-Efficiency Targeting**:
   Our tuned LightGBM model captures **51.6% of all churners in the top 20% highest-risk pool** (Recall@20%), with a **2.73x Top-Decile Lift** and **0.8412 ROC-AUC** [1,000x Bootstrap 95% CI: 0.8155, 0.8679].

3. 🔍 **Frontline Explainability**:
   Real-time TreeSHAP generates top-3 plain-language reason codes per customer (e.g. "Month-to-month contract on Fiber without TechSupport add-on") directly empowering call center agents.

4. 🚀 **Production Architecture**:
   - FastAPI REST API with p95 latency of **3.27 ms** (< 100 ms SLA).
   - Containerized with Docker, automated CI/CD quality gate (85% coverage, ruff).
   - Automated Data Drift surveillance using Evidently AI with retrain trigger alerts.
   - Interactive Streamlit decision cockpit.

🔗 **GitHub (Source & Architecture)**: https://github.com/ZeeqRyz/02_ChurnGuard_Telco_Churn_Prediction
🌐 **Live Interactive Demo**: [Hugging Face Spaces URL]
📖 **Kaggle Notebook**: [Kaggle URL]

Stack: Python | Scikit-Learn | LightGBM | Optuna | SHAP | MLflow | FastAPI | Docker | GitHub Actions | Evidently AI | Streamlit

#MachineLearning #DataScience #MLOps #Telco #CustomerRetention #FastAPI #LightGBM #MalaysiaTech #Portfolio
```

---

### CV / Resume Bullet Points

- **Machine Learning Engineer — ChurnGuard Telco Retention Platform**
  - Designed and deployed an end-to-end cost-aware churn prediction system for a telco cohort (7,043 customers), achieving **0.8412 ROC-AUC** and **2.73x Top-Decile Lift** using tuned LightGBM and Isotonic calibration (Brier score 0.1381).
  - Maximized expected campaign profit by developing a business cost matrix ($\tau^*=0.18$), boosting net campaign ROI by **+44.2% (RM 35,206 per 1k customers)** over standard cutoffs.
  - Deployed containerized FastAPI REST service delivering **3.27 ms p95 inference latency** with real-time SHAP explainability reason codes, backed by GitHub Actions CI/CD (85% test coverage) and Evidently AI drift monitoring.
