"""
SupplyGuard AI - Preprocessing Module
======================================
Cleans the raw dataset: drops duplicates, imputes missing values,
fixes dtypes, and returns an analysis-ready dataframe.

All paths are built with os.path.join relative to the project root so this
module (and anything that imports it, e.g. the dashboard) runs unmodified on
Windows, macOS, and Linux.
"""

import os
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE, "data", "global_supply_chain_risk_logistics.csv")
CLEAN_PATH = os.path.join(BASE, "data", "clean_supply_chain_data.csv")           # kept for backward compatibility
PREPROCESSED_PATH = os.path.join(BASE, "data", "preprocessed_data.csv")          # canonical name used by the dashboard


def load_raw(path=RAW_PATH):
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame):
    """Cleans a raw dataframe and returns (clean_df, meta) where meta carries
    full before/after data-quality stats (rows, columns, missing values, duplicates)
    for display in the dashboard's Preprocessing & EDA section."""
    df = df.copy()

    rows_before = len(df)
    columns_before = df.shape[1]
    missing_before_total = int(df.isna().sum().sum())
    duplicate_rows_before = int(df.duplicated().sum())

    # 1. Drop exact duplicate rows (keep first) — ignore Shipment_ID since it's a unique key
    #    and wouldn't by itself make two otherwise-identical rows look different.
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "Shipment_ID"], keep="first")
    dupes_removed = rows_before - len(df)

    # 2. Fix Shipment_ID duplicates from shuffling (re-index safely)
    df = df.reset_index(drop=True)

    # 3. Impute missing numeric values with median (robust to outliers)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    missing_report = df[numeric_cols].isna().sum()
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # 4. Impute missing categoricals with mode
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in cat_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 5. Clip obviously invalid values (defensive data-quality checks)
    if "On_Time_Delivery_Rate_Pct" in df.columns:
        df["On_Time_Delivery_Rate_Pct"] = df["On_Time_Delivery_Rate_Pct"].clip(0, 100)
    if "Demand_Variability_Pct" in df.columns:
        df["Demand_Variability_Pct"] = df["Demand_Variability_Pct"].clip(0, 100)
    for col in ["Supplier_Reliability_Score", "Financial_Stability_Score", "Geopolitical_Risk_Index",
                "Weather_Risk_Index", "Port_Congestion_Index", "Fuel_Price_Index",
                "Contract_Flexibility_Score", "Latent_Risk_Score"]:
        if col in df.columns:
            df[col] = df[col].clip(0, 1)
    if "Lead_Time_Days" in df.columns:
        df["Lead_Time_Days"] = df["Lead_Time_Days"].clip(lower=1)
    if "Customs_Delay_Days" in df.columns:
        df["Customs_Delay_Days"] = df["Customs_Delay_Days"].clip(lower=0)

    meta = {
        "rows_before": rows_before,
        "rows_after": len(df),
        "columns_before": columns_before,
        "columns_after": df.shape[1],
        "missing_before": missing_report[missing_report > 0].to_dict(),
        "missing_values_before_total": missing_before_total,
        "missing_values_after_total": int(df.isna().sum().sum()),
        "duplicate_rows_before": duplicate_rows_before,
        "duplicates_removed": int(dupes_removed),
        "duplicate_rows_after": int(df.duplicated().sum()),
    }
    return df, meta


def save_preprocessed(clean_df: pd.DataFrame):
    """Writes the cleaned dataframe to both the canonical preprocessed_data.csv
    (used by the dashboard) and the legacy clean_supply_chain_data.csv path
    (still used by feature_engineering.py), so nothing downstream breaks."""
    os.makedirs(os.path.dirname(PREPROCESSED_PATH), exist_ok=True)
    clean_df.to_csv(PREPROCESSED_PATH, index=False)
    clean_df.to_csv(CLEAN_PATH, index=False)


if __name__ == "__main__":
    raw = load_raw()
    clean, meta = preprocess(raw)
    print(meta)
    save_preprocessed(clean)
    print("Clean shape:", clean.shape)
    print("Saved to:", PREPROCESSED_PATH, "and", CLEAN_PATH)
