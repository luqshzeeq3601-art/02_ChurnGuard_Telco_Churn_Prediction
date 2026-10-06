"""Streamlit interactive dashboard for ChurnGuard.

Features:
1. Executive KPIs & Overview: Model metrics and Malaysian telco framing (NusaTel).
2. Single Customer Assessment: Interactive profile builder, probability gauge, risk tiers, and SHAP reasons.
3. Batch Scoring & Campaign Targeting: CSV upload, ranked retention list, and campaign profit simulator.
4. Strategic Business Insights: Interactive churn segment deep-dives and profit curve.
5. Malaysia Telecom Market Trends: data.gov.my historical subscriber analytics.
6. MLOps Monitoring & Drift: Evidently data drift overview, retrain triggers, and drift report link.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from churnguard.config import CFG
from churnguard.dashboard import load_dashboard_summary
from churnguard.models.predict import ChurnPredictor

# Page setup
st.set_page_config(
    page_title="ChurnGuard | Telco Retention Cockpit",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for executive styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .badge-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
    }
    .badge-medium {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
    }
    .badge-low {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_predictor() -> ChurnPredictor:
    """Load model pipeline and SHAP explainer once."""
    return ChurnPredictor()


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    """Load sample test dataset for quick demos."""
    test_path = CFG["paths"]["processed_dir"] / "test.parquet"
    if test_path.exists():
        return pd.read_parquet(test_path)
    csv_path = CFG["paths"]["processed_dir"] / "test.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


@st.cache_data
def load_metrics_meta() -> dict:
    """Load final model metrics and metadata."""
    meta_path = Path(CFG["paths"]["models_dir"]) / "model_meta.json"
    return load_dashboard_summary(meta_path, Path("reports/final_metrics.json"))


@st.cache_data
def load_drift_summary() -> dict:
    """Load latest Evidently drift summary."""
    drift_json = Path(CFG["paths"]["reports_dir"]) / "drift" / "drift_summary.json"
    if drift_json.exists():
        with open(drift_json, encoding="utf-8") as f:
            return json.load(f)
    return {}


predictor = get_predictor()
meta = load_metrics_meta()
sample_df = load_sample_data()
drift_summary = load_drift_summary()


# Sidebar Navigation & System Meta
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/Python-Dark.svg",
        width=40,
    )
    st.title("ChurnGuard AI")
    st.caption("Telco Retention Targeting Engine")
    st.markdown("---")

    selected_page = st.radio(
        "Navigation",
        [
            "🎯 Single Customer Assessment",
            "📊 Batch Scoring & Retention Targeting",
            "💡 Strategic Business Insights",
            "🇲🇾 Malaysia Market Context",
            "📡 Drift & Health Monitoring",
        ],
    )

    st.markdown("---")
    st.subheader("Model Status")
    st.write(f"**Model:** `{meta.get('model_name', 'Unknown model')}`")
    st.write(f"**Version:** `{meta.get('model_version', '1.0.0')}`")
    st.write(f"**Optimal Threshold ($\\tau^*$):** `{predictor.optimal_threshold:.4f}`")
    recorded = meta.get("evaluation_metrics", {})
    for label, key in [("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc")]:
        value = recorded.get(key, {}).get("point_estimate")
        display = f"{value:.4f}" if value is not None else "unavailable"
        st.write(f"**{label} (Recorded test):** `{display}`")
    st.caption("Framed for fictional Malaysian telco **NusaTel** (RM currency).")


# Header
st.markdown(
    '<div class="main-header">🛡️ ChurnGuard: Telco Churn Intelligence Platform</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Production-grade ML decision engine for cost-aware customer retention, SHAP frontline explanations, and proactive intervention targeting.</div>',
    unsafe_allow_html=True,
)


# ==========================================
# PAGE 1: Single Customer Assessment
# ==========================================
if selected_page == "🎯 Single Customer Assessment":
    st.subheader("Single Customer Risk Profiler & Intervention Advisor")
    st.info(
        "💡 Adjust customer profile attributes below to calculate the calibrated churn probability, determine the risk tier, and view plain-language frontline explanation reason codes."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 👤 Demographics & Identity")
        customer_id = st.text_input("Customer ID", value="CUST-MY-8921")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox(
            "Senior Citizen", [0, 1], format_func=lambda x: "Yes (>= 60 yo)" if x == 1 else "No"
        )
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (Months with NusaTel)", min_value=0, max_value=72, value=3)

    with col2:
        st.markdown("##### 🌐 Subscribed Services")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    with col3:
        st.markdown("##### 💳 Contract & Billing")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = st.number_input(
            "Monthly Charges (RM)", min_value=15.0, max_value=150.0, value=89.50, step=1.0
        )
        est_total = max(monthly_charges, round(monthly_charges * max(1, tenure), 2))
        total_charges = st.number_input(
            "Total Charges (RM)",
            min_value=0.0,
            max_value=12000.0,
            value=float(est_total),
            step=10.0,
        )

    customer_payload = {
        "customerID": customer_id,
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    if st.button("🚀 Evaluate Customer Risk", type="primary", use_container_width=True):
        res = predictor.predict_single(customer_payload)
        prob = res["churn_probability"]
        tier = res["risk_tier"]
        reasons = res["top_reasons"]

        st.markdown("---")
        st.markdown("### Risk Evaluation Results")

        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Calibrated Churn Probability", f"{prob:.1%}")
        with r2:
            badge_class = f"badge-{tier.lower()}"
            st.markdown(
                f"**Assigned Risk Tier:**<br><span class='{badge_class}'>{tier.upper()} RISK</span>",
                unsafe_allow_html=True,
            )
        with r3:
            if tier == "High":
                rec = "🎯 Proactive RM50 voucher / Contract upgrade offer"
            elif tier == "Medium":
                rec = "📩 Digital engagement / Loyalty survey"
            else:
                rec = "✅ Organic service / Cross-sell opportunities"
            st.markdown(f"**Recommended Action:**<br>{rec}", unsafe_allow_html=True)

        st.markdown("#### 🔍 Plain-Language Frontline Explanation Codes (SHAP)")
        for i, reason in enumerate(reasons, 1):
            st.markdown(f"**{i}.** {reason}")


# ==========================================
# PAGE 2: Batch Scoring & Retention Targeting
# ==========================================
elif selected_page == "📊 Batch Scoring & Retention Targeting":
    st.subheader("Batch Scoring & Profit-Optimized Campaign Targeting")
    st.markdown(
        "Upload customer records to rank churn risks and simulate expected retention campaign ROI."
    )

    upload_file = st.file_uploader("Upload Customer CSV / Parquet", type=["csv", "parquet"])

    use_sample = st.checkbox(
            "Use public IBM Telco benchmark cohort (1,057 rows)", value=(upload_file is None)
    )

    df_to_score = None
    if upload_file is not None:
        if upload_file.name.endswith(".parquet"):
            df_to_score = pd.read_parquet(upload_file)
        else:
            df_to_score = pd.read_csv(upload_file)
    elif use_sample and len(sample_df) > 0:
        df_to_score = sample_df.copy()

    if df_to_score is not None:
        st.write(f"Loaded **{len(df_to_score):,}** customer records.")

        with st.expander("⚙️ Campaign Strategy & Budget Constraints", expanded=True):
            strat_col1, strat_col2 = st.columns([2, 1])
            with strat_col1:
                target_strategy = st.radio(
                    "Retention Campaign Strategy",
                    [
                        "🎯 Profit-Optimal (Unconstrained, tau=0.1882) — Maximize net RM return (~49.8% contact rate, 89.0% recall)",
                        "⚖️ Balanced Capacity (Top 30% Cap, tau=0.3667) — Conserve retention vouchers (~30.0% contact rate, 66.2% recall)",
                        "⚡ Strict Budget (Top 20% Cap, tau=0.4714) — Focus frontline call-center capacity (~20.1% contact rate, 48.4% recall)",
                        "🛠️ Custom Risk Cutoff",
                    ],
                    index=0,
                )
            with strat_col2:
                if "Custom Risk Cutoff" in target_strategy:
                    custom_tau = st.slider(
                        "Custom Decision Threshold (tau)", 0.05, 0.95, 0.1882, 0.01
                    )
                else:
                    custom_tau = None

            c_col1, c_col2, c_col3 = st.columns(3)
            with c_col1:
                offer_cost = st.number_input("Retention Offer Cost (RM)", value=50.0, step=5.0)
            with c_col2:
                retention_rate = st.slider("Retention Acceptance Rate (%)", 10, 80, 30) / 100.0
            with c_col3:
                clv_retained = st.number_input("CLV if Retained (RM)", value=780.0, step=50.0)

        if st.button("⚡ Score & Rank Entire Batch", type="primary"):
            with st.spinner("Executing model scoring and SHAP reason extraction..."):
                scored_df = predictor.predict_batch(df_to_score, include_reasons=True)
                n_total = len(scored_df)

                # Determine active threshold from chosen strategy
                if "Top 30%" in target_strategy:
                    active_tau = 0.3667
                    strategy_label = "Balanced Capacity (Top 30% Cap)"
                elif "Top 20%" in target_strategy:
                    active_tau = 0.4714
                    strategy_label = "Strict Budget (Top 20% Cap)"
                elif custom_tau is not None:
                    active_tau = custom_tau
                    strategy_label = f"Custom (tau={active_tau:.2f})"
                else:
                    active_tau = 0.1882
                    strategy_label = "Profit-Optimal (Unconstrained)"

                # Targeted flag based on active strategy
                scored_df["targeted_in_campaign"] = scored_df["churn_probability"] >= active_tau

                # Summary metrics
                n_targeted = int(scored_df["targeted_in_campaign"].sum())
                n_high = (scored_df["risk_tier"] == "High").sum()
                n_med = (scored_df["risk_tier"] == "Medium").sum()
                n_low = (scored_df["risk_tier"] == "Low").sum()

                # Estimated campaign economics
                exp_saved = n_targeted * retention_rate
                gross_clv_saved = exp_saved * clv_retained
                campaign_cost = n_targeted * offer_cost
                net_campaign_profit = gross_clv_saved - campaign_cost

                st.markdown("---")
                st.markdown(f"### 📈 Campaign Economics Summary ({strategy_label})")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Cohort", f"{n_total:,}")
                m2.metric("Contacted (Targeted)", f"{n_targeted:,} ({n_targeted/n_total:.1%})")
                m3.metric("Estimated Voucher Outlay", f"RM {campaign_cost:,.2f}")
                m4.metric("Net Projected Campaign Profit", f"RM {net_campaign_profit:,.2f}")

                st.markdown("### 📋 Ranked Customer Priority List")
                tier_filter = st.multiselect(
                    "Filter by Risk Tier",
                    ["High", "Medium", "Low"],
                    default=["High", "Medium"],
                )
                filtered_df = scored_df[scored_df["risk_tier"].isin(tier_filter)]

                display_cols = [
                    "rank",
                    "customerID",
                    "churn_probability",
                    "risk_tier",
                    "tenure",
                    "Contract",
                    "MonthlyCharges",
                    "top_reasons",
                ]
                existing_cols = [c for c in display_cols if c in filtered_df.columns]
                st.dataframe(filtered_df[existing_cols].head(100), use_container_width=True)

                # CSV Download
                csv_buffer = io.StringIO()
                filtered_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Scored Retention List (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="churnguard_scored_campaign.csv",
                    mime="text/csv",
                )


# ==========================================
# PAGE 3: Strategic Business Insights
# ==========================================
elif selected_page == "💡 Strategic Business Insights":
    st.subheader("Key Drivers & Strategic Retention Analytics")

    tab1, tab2, tab3 = st.tabs(
        ["📊 Churn Risk Drivers", "💰 Profit Curve Optimization", "⚖️ Fairness & Parity"]
    )

    with tab1:
        st.markdown("#### Top Empirical Drivers of Churn")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 1. Contract Lock-in Effect")
            st.markdown(
                "Month-to-month contracts exhibit a **42.9% churn rate** (88.6% of all churners), compared to **10.9%** for 1-year and **3.0%** for 2-year commitments."
            )
            fig1, ax1 = plt.subplots(figsize=(6, 3.5))
            contract_data = pd.DataFrame(
                {
                    "Contract": ["Month-to-month", "One year", "Two year"],
                    "Churn Rate (%)": [42.9, 10.9, 3.0],
                }
            )
            sns.barplot(
                data=contract_data,
                x="Contract",
                y="Churn Rate (%)",
                palette=["#EF4444", "#3B82F6", "#10B981"],
                ax=ax1,
            )
            ax1.set_ylim(0, 50)
            st.pyplot(fig1)

        with c2:
            st.markdown("##### 2. Fiber Optic Service Deficit")
            st.markdown(
                "Fiber optic customers churn at **41.8%** vs **19.2%** for DSL, escalating to **>48%** when Tech Support is absent. Packaging support bundles is critical."
            )
            fig2, ax2 = plt.subplots(figsize=(6, 3.5))
            service_data = pd.DataFrame(
                {
                    "Internet Service": ["DSL", "Fiber optic", "No Internet"],
                    "Churn Rate (%)": [19.2, 41.8, 7.4],
                }
            )
            sns.barplot(
                data=service_data,
                x="Internet Service",
                y="Churn Rate (%)",
                palette=["#3B82F6", "#EF4444", "#94A3B8"],
                ax=ax2,
            )
            ax2.set_ylim(0, 50)
            st.pyplot(fig2)

    with tab2:
        st.markdown("#### Optimal Campaign Decision Threshold ($\tau^* = 0.18$)")
        st.markdown(
            """
            Classical ML defaults ($\tau = 0.5$) severely under-target churners due to asymmetric business costs.
            By maximizing expected profit per 1,000 customers:
            - **Default $\\tau = 0.50$:** RM 27,510 expected campaign profit per 1k customers.
            - **Optimal $\\tau^* = 0.18$:** **RM 39,678 expected campaign profit per 1k customers (+44.2% gain)**.
            """
        )
        profit_fig = Path(CFG["paths"]["figures_dir"]) / "08_profit_curve.png"
        if profit_fig.exists():
            st.image(str(profit_fig), caption="Campaign Profit Curve vs Decision Threshold")

    with tab3:
        st.markdown("#### Algorithmic Fairness & Demographic Parity")
        st.markdown(
            r"""
            - **Gender Parity:** Female Recall = **82.3%**, Male Recall = **83.5%** (Gap = **0.0115** $\le 0.05 \implies$ Passed NFR7).
            - **Senior Citizen Parity:** Senior Recall = **93.7%**, Non-Senior Recall = **78.6%** (Higher recall for seniors reflects concentrated fiber optic plan adoption).
            - **Ethical Safeguard:** No individual is penalized based on protected demographic attributes.
            """
        )


# ==========================================
# PAGE 4: Malaysia Market Context
# ==========================================
elif selected_page == "🇲🇾 Malaysia Market Context":
    st.subheader("Malaysian Telecom Market Dynamics (data.gov.my)")
    st.markdown(
        """
        In Malaysia's maturing mobile market (>50 million cellular subscriptions, >140% penetration), net new subscriber growth has flattened.
        Revenue growth is strictly driven by **ARPU expansion** and **reducing postpaid customer churn**.
        """
    )

    my_fig = Path(CFG["paths"]["figures_dir"]) / "06_malaysia_cellular_trends.png"
    if my_fig.exists():
        st.image(
            str(my_fig),
            caption="Historical Cellular Subscriptions in Malaysia (2000-2021) - data.gov.my",
        )

    st.markdown(
        """
        ##### Strategic Implications for NusaTel:
        1. **Postpaid Migration:** Postpaid subscribers represent the highest-margin segment (>RM65/mo ARPU). Preventing 10% churn in postpaid yields disproportionate enterprise profitability.
        2. **Onboarding Guardrail:** 53.1% of first-half year month-to-month subscribers churn. NusaTel must deploy onboarding retention plays within the first 90 days.
        """
    )


# ==========================================
# PAGE 5: Drift & Health Monitoring
# ==========================================
elif selected_page == "📡 Drift & Health Monitoring":
    st.subheader("MLOps Health & Data Drift Surveillance (Evidently AI)")
    st.markdown(
        "Automated drift detection compares current production batches against the baseline training distribution to recommend model retraining."
    )

    if drift_summary:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric(
            "Dataset Drift Detected",
            "No" if not drift_summary.get("dataset_drift_detected") else "YES",
        )
        d2.metric("Drift Share", f"{drift_summary.get('share_of_drifted_columns', 0.0):.1%}")
        d3.metric(
            "Drifted Features",
            f"{drift_summary.get('number_of_drifted_columns', 0)} / {drift_summary.get('total_columns_analyzed', 0)}",
        )
        retrain_flag = drift_summary.get("retrain_recommended", False)
        d4.metric("Retrain Alert", "HEALTHY" if not retrain_flag else "RETRAIN RECOMMENDED")

        st.markdown("#### Drifted Feature Inventory")
        st.write(drift_summary.get("drifted_features", []))

        html_report_path = Path(CFG["paths"]["reports_dir"]) / "drift" / "drift_report.html"
        if html_report_path.exists():
            with open(html_report_path, encoding="utf-8") as f:
                html_data = f.read()
            st.download_button(
                label="📥 Download Full Interactive Evidently HTML Report",
                data=html_data,
                file_name="churnguard_evidently_drift_report.html",
                mime="text/html",
            )
            st.info(
                "Full interactive drift report generated and available at `reports/drift/drift_report.html`."
            )
    else:
        st.warning(
            "No drift summary found. Run `make drift` to generate the latest Evidently report."
        )
