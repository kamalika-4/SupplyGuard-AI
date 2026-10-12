"""
SupplyGuard AI - Exploratory Data Analysis
=============================================
Generates key EDA charts as PNGs (used in reports) — the live dashboard
renders interactive versions with Plotly directly from data.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="viridis")
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE, "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
df = pd.read_csv(os.path.join(BASE, "data", "feature_engineered_data.csv"))

# 1. Risk level distribution
plt.figure(figsize=(6, 4))
order = ["Low", "Medium", "High"]
sns.countplot(data=df, x="Risk_Level", order=order, palette=["#2ecc71", "#f39c12", "#e74c3c"])
plt.title("Risk Level Distribution")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_risk_level_distribution.png", dpi=120)
plt.close()

# 2. Risk by region
plt.figure(figsize=(9, 5))
region_risk = pd.crosstab(df["Region"], df["Risk_Level"], normalize="index")[order] * 100
region_risk.sort_values("High", ascending=False).plot(
    kind="bar", stacked=True, color=["#2ecc71", "#f39c12", "#e74c3c"], figsize=(9, 5))
plt.ylabel("% of Shipments")
plt.title("Risk Composition by Region")
plt.legend(title="Risk Level")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_risk_by_region.png", dpi=120)
plt.close()

# 3. Correlation heatmap of key numeric drivers
key_cols = ["Geopolitical_Risk_Index", "Weather_Risk_Index", "Port_Congestion_Index",
            "Supplier_Reliability_Score", "Financial_Stability_Score", "Customs_Delay_Days",
            "Disruption_Events_Past_Year", "On_Time_Delivery_Rate_Pct", "Cost_Variance_Pct",
            "Early_Warning_Score"]
plt.figure(figsize=(9, 7))
sns.heatmap(df[key_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap - Key Risk Drivers")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_correlation_heatmap.png", dpi=120)
plt.close()

# 4. Early Warning Score distribution by risk level
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Risk_Level", y="Early_Warning_Score", order=order,
            palette=["#2ecc71", "#f39c12", "#e74c3c"])
plt.title("Early Warning Score by Risk Level")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_ews_by_risk_level.png", dpi=120)
plt.close()

# 5. Transport mode vs disruption events
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Transport_Mode", y="Disruption_Events_Past_Year",
            hue="Transport_Mode", legend=False, estimator="mean")
plt.title("Avg Disruption Events by Transport Mode")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_transport_mode_disruptions.png", dpi=120)
plt.close()

# 6. Product category risk
plt.figure(figsize=(10, 5))
cat_order = df.groupby("Product_Category")["Early_Warning_Score"].mean().sort_values(ascending=False).index
sns.barplot(data=df, x="Product_Category", y="Early_Warning_Score", order=cat_order,
            hue="Product_Category", legend=False)
plt.xticks(rotation=35, ha="right")
plt.title("Avg Early Warning Score by Product Category")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/06_product_category_ews.png", dpi=120)
plt.close()

print("EDA figures saved to", FIG_DIR)
