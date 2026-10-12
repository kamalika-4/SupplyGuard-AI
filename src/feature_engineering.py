"""
SupplyGuard AI - Feature Engineering Module
=============================================
Creates domain-driven derived features that power the Early Warning Score
and Explainable Risk Factors in the dashboard.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Composite risk sub-scores (0-100 scale, higher = riskier) ---
    df["Geo_Weather_Exposure"] = (
        0.55 * df["Geopolitical_Risk_Index"] + 0.45 * df["Weather_Risk_Index"]
    ) * 100

    df["Supplier_Weakness_Score"] = (
        0.5 * (1 - df["Supplier_Reliability_Score"]) +
        0.5 * (1 - df["Financial_Stability_Score"])
    ) * 100

    df["Logistics_Friction_Score"] = (
        0.4 * df["Port_Congestion_Index"] +
        0.3 * np.clip(df["Customs_Delay_Days"] / 10, 0, 1) +
        0.3 * np.clip(df["Lead_Time_Days"] / 45, 0, 1)
    ) * 100

    df["Demand_Volatility_Score"] = (
        0.6 * (df["Demand_Variability_Pct"] / 100) +
        0.4 * (1 - np.clip(df["Inventory_Level_Days"] / 30, 0, 1))
    ) * 100

    df["Concentration_Risk_Score"] = (
        0.7 * df["Single_Sourcing_Flag"] + 0.3 * (1 - df["Contract_Flexibility_Score"])
    ) * 100

    df["Historical_Instability_Score"] = np.clip(
        (df["Disruption_Events_Past_Year"] / 8) * 60 + (100 - df["On_Time_Delivery_Rate_Pct"]) * 0.4, 0, 100)

    df["Cost_Pressure_Score"] = np.clip(
        (df["Cost_Variance_Pct"] + 10) / 70 * 100 * 0.6 + df["Fuel_Price_Index"] * 40, 0, 100)

    # --- Weighted Early Warning Score (0-100), transparent & explainable ---
    weights = {
        "Geo_Weather_Exposure": 0.20,
        "Supplier_Weakness_Score": 0.18,
        "Logistics_Friction_Score": 0.16,
        "Demand_Volatility_Score": 0.12,
        "Concentration_Risk_Score": 0.12,
        "Historical_Instability_Score": 0.14,
        "Cost_Pressure_Score": 0.08,
    }
    raw_score = sum(df[k] * w for k, w in weights.items())
    # Min-max stretch to fully utilize the 0-100 scale so the score is interpretable
    lo, hi = raw_score.min(), raw_score.max()
    df["Early_Warning_Score"] = (((raw_score - lo) / (hi - lo)) * 100).round(1)

    # Quantile-anchored bands (kept consistent with overall risk distribution: ~45/35/13/7)
    q45, q80, q93 = df["Early_Warning_Score"].quantile([0.45, 0.80, 0.93])

    def ews_band(x):
        if x < q45:
            return "Safe"
        elif x < q80:
            return "Watch"
        elif x < q93:
            return "Elevated"
        return "Critical"
    df["EWS_Band"] = df["Early_Warning_Score"].apply(ews_band)

    # --- Encode categoricals for ML (keep originals for dashboard display) ---
    cat_cols = ["Region", "Country", "Product_Category", "Supplier_Tier", "Transport_Mode"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders, weights


FEATURE_WEIGHTS_DOC = """
Early Warning Score (EWS) formula — fully transparent, 0-100 scale:

EWS = 0.20*Geo_Weather_Exposure + 0.18*Supplier_Weakness_Score
    + 0.16*Logistics_Friction_Score + 0.12*Demand_Volatility_Score
    + 0.12*Concentration_Risk_Score + 0.14*Historical_Instability_Score
    + 0.08*Cost_Pressure_Score

Bands: Safe (<35) | Watch (35-54) | Elevated (55-74) | Critical (75-100)
"""

if __name__ == "__main__":
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean_path = os.path.join(base, "data", "clean_supply_chain_data.csv")
    fe_path = os.path.join(base, "data", "feature_engineered_data.csv")
    df = pd.read_csv(clean_path)
    fe_df, encoders, weights = engineer_features(df)
    fe_df.to_csv(fe_path, index=False)
    print("Feature engineered shape:", fe_df.shape)
    print(fe_df["EWS_Band"].value_counts())
    print(FEATURE_WEIGHTS_DOC)
