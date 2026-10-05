"""Generate high-resolution EDA figures and save them to reports/figures/.

Adheres to docs/05_DATA_SPEC.md section 6:
- Target distribution
- Churn rate by each categorical
- Numeric distributions by churn
- Cramer's V / correlation heatmap
- Churn by tenure bucket x contract heatmap
- Malaysia postpaid vs prepaid trend (data.gov.my)
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

from churnguard.config import CFG
from churnguard.data.load import load_malaysia_subscribers


def compute_cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Compute Cramer's V statistic for categorical-categorical association."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2corr / denom))


def generate_all_eda_figures(output_dir: Path | str | None = None) -> list[Path]:
    """Generate and save all 6 EDA figures per 05_DATA_SPEC.md."""
    out_path = Path(output_dir) if output_dir else CFG["paths"]["figures_dir"]
    out_path.mkdir(parents=True, exist_ok=True)

    # Use train split to prevent data leakage during exploratory analysis
    train_path = CFG["paths"]["processed_dir"] / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Training split not found at {train_path}. Run split.py first.")

    train_df = pd.read_parquet(train_path)

    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

    saved_files: list[Path] = []

    # -------------------------------------------------------------
    # 1. Target Distribution
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = train_df["Churn"].value_counts()
    percentages = train_df["Churn"].value_counts(normalize=True) * 100
    labels = ["Retained (0)", "Churned (1)"]
    colors = ["#2ecc71", "#e74c3c"]

    bars = ax.bar(labels, counts, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    for bar, pct in zip(bars, percentages):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 60,
            f"{int(yval):,} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_title("Target Distribution (Training Set: N = 4,930)", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Customer Count", fontsize=12)
    ax.set_ylim(0, max(counts) * 1.15)
    f1 = out_path / "01_target_distribution.png"
    plt.savefig(f1, dpi=300)
    plt.close()
    saved_files.append(f1)

    # -------------------------------------------------------------
    # 2. Key Categorical Churn Rates
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    cat_features = [
        ("Contract", "Contract Type"),
        ("InternetService", "Internet Service"),
        ("TechSupport", "Tech Support"),
        ("OnlineSecurity", "Online Security"),
        ("PaymentMethod", "Payment Method"),
        ("PaperlessBilling", "Paperless Billing"),
    ]

    for idx, (col, title) in enumerate(cat_features):
        ax = axes[idx // 3, idx % 3]
        rate_df = (
            train_df.groupby(col)["Churn"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "churn_rate"})
        )
        rate_df["churn_rate_pct"] = rate_df["churn_rate"] * 100

        bars = ax.bar(
            rate_df[col],
            rate_df["churn_rate_pct"],
            color="#3498db",
            edgecolor="black",
            linewidth=1.0,
        )
        ax.axhline(
            train_df["Churn"].mean() * 100,
            color="#e74c3c",
            linestyle="--",
            label="Baseline (26.5%)",
            alpha=0.8,
        )

        for bar, row in zip(bars, rate_df.itertuples()):
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + 1.0,
                f"{yval:.1f}%\n(n={row.count})",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_title(f"Churn Rate by {title}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Churn Rate (%)", fontsize=10)
        ax.set_ylim(0, max(rate_df["churn_rate_pct"]) * 1.25)
        ax.tick_params(axis="x", rotation=25)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=9)

    plt.suptitle("Churn Rates across Key Categorical Segments", fontsize=16, fontweight="bold", y=1.02)
    f2 = out_path / "02_categorical_churn_rates.png"
    plt.savefig(f2, dpi=300)
    plt.close()
    saved_files.append(f2)

    # -------------------------------------------------------------
    # 3. Numeric Distributions by Churn (tenure, MonthlyCharges, TotalCharges)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    num_features = [
        ("tenure", "Tenure (Months)"),
        ("MonthlyCharges", "Monthly Charges (RM)"),
        ("TotalCharges", "Total Charges (RM)"),
    ]

    for idx, (col, label) in enumerate(num_features):
        ax = axes[idx]
        sns.boxplot(
            data=train_df,
            x="Churn",
            y=col,
            hue="Churn",
            legend=False,
            ax=ax,
            palette=["#2ecc71", "#e74c3c"],
            showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"},
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Retained (0)", "Churned (1)"])
        ax.set_title(f"{label} by Churn", fontsize=12, fontweight="bold")
        ax.set_xlabel("Status", fontsize=11)
        ax.set_ylabel(label, fontsize=11)

    plt.suptitle("Numeric Feature Distributions by Churn Status", fontsize=15, fontweight="bold", y=1.03)
    f3 = out_path / "03_numeric_distributions.png"
    plt.savefig(f3, dpi=300)
    plt.close()
    saved_files.append(f3)

    # -------------------------------------------------------------
    # 4. Association / Cramer's V Matrix
    # -------------------------------------------------------------
    all_cats = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "Churn",
    ]

    cramers_matrix = pd.DataFrame(index=all_cats, columns=all_cats, dtype=float)
    for c1 in all_cats:
        for c2 in all_cats:
            if c1 == c2:
                cramers_matrix.loc[c1, c2] = 1.0
            else:
                cramers_matrix.loc[c1, c2] = compute_cramers_v(train_df[c1], train_df[c2])

    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(
        cramers_matrix.astype(float),
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        cbar_kws={"label": "Cramer's V Association"},
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Categorical Feature Associations (Cramer's V)", fontsize=15, fontweight="bold", pad=15)
    f4 = out_path / "04_cramers_v_association.png"
    plt.savefig(f4, dpi=300)
    plt.close()
    saved_files.append(f4)

    # -------------------------------------------------------------
    # 5. Churn by Tenure Bucket x Contract Type Heatmap
    # -------------------------------------------------------------
    bins = [-1, 6, 12, 24, 48, 72]
    bin_labels = ["0-6m", "7-12m", "13-24m", "25-48m", "49-72m"]
    train_df_copy = train_df.copy()
    train_df_copy["tenure_bucket"] = pd.cut(train_df_copy["tenure"], bins=bins, labels=bin_labels)

    heatmap_data = (
        train_df_copy.pivot_table(
            index="Contract",
            columns="tenure_bucket",
            values="Churn",
            aggfunc="mean",
            observed=False,
        )
        * 100
    )

    counts_data = train_df_copy.pivot_table(
        index="Contract",
        columns="tenure_bucket",
        values="Churn",
        aggfunc="count",
        observed=False,
    )

    # Reorder index for clarity
    contract_order = ["Month-to-month", "One year", "Two year"]
    heatmap_data = heatmap_data.reindex(contract_order)
    counts_data = counts_data.reindex(contract_order)

    annot_matrix = np.empty(heatmap_data.shape, dtype=object)
    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            val = heatmap_data.iloc[i, j]
            cnt = counts_data.iloc[i, j]
            annot_matrix[i, j] = f"{val:.1f}%\n(n={cnt})"

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        heatmap_data,
        annot=annot_matrix,
        fmt="",
        cmap="Reds",
        cbar_kws={"label": "Churn Rate (%)"},
        ax=ax,
        linewidths=1.0,
    )
    ax.set_title("Churn Rate by Contract Type × Tenure Bucket", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Contract Type", fontsize=12)
    ax.set_xlabel("Tenure Bucket", fontsize=12)
    f5 = out_path / "05_tenure_contract_heatmap.png"
    plt.savefig(f5, dpi=300)
    plt.close()
    saved_files.append(f5)

    # -------------------------------------------------------------
    # 6. Malaysia Postpaid vs Prepaid Trend (data.gov.my)
    # -------------------------------------------------------------
    my_df = load_malaysia_subscribers()
    my_df["date"] = pd.to_datetime(my_df["date"])
    my_df["year"] = my_df["date"].dt.year
    my_pivot = my_df.pivot(index="year", columns="plan", values="subscriptions") / 1_000_000

    fig, ax = plt.subplots(figsize=(10, 6))
    if "postpaid" in my_pivot.columns:
        ax.plot(my_pivot.index, my_pivot["postpaid"], marker="o", linewidth=2.5, label="Postpaid", color="#2980b9")
    if "prepaid" in my_pivot.columns:
        ax.plot(my_pivot.index, my_pivot["prepaid"], marker="s", linewidth=2.5, label="Prepaid", color="#e67e22")
    if "total" in my_pivot.columns:
        ax.plot(my_pivot.index, my_pivot["total"], marker="^", linewidth=2.0, linestyle="--", label="Total", color="#2c3e50")

    ax.set_title("Malaysia Cellular Subscriptions Trend (2000 - 2021, data.gov.my)", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Subscriptions (Millions)", fontsize=12)
    ax.set_xlabel("Year", fontsize=12)
    ax.legend(fontsize=11)
    ax.set_xticks(my_pivot.index[::2])
    f6 = out_path / "06_malaysia_cellular_trends.png"
    plt.savefig(f6, dpi=300)
    plt.close()
    saved_files.append(f6)

    return saved_files


if __name__ == "__main__":
    saved = generate_all_eda_figures()
    print(f"Generated {len(saved)} figures:")
    for f in saved:
        print(f" - {f}")
