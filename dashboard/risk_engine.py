"""
SupplyGuard AI - Risk Engine
=============================
Shared logic for the Predict -> Explain -> Recommend workflow used across
the Streamlit dashboard (live prediction form, What-if Simulator, Risk Ranking).
"""

import numpy as np
import pandas as pd
import joblib
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, "models")

EWS_WEIGHTS = {
    "Geo_Weather_Exposure": 0.20,
    "Supplier_Weakness_Score": 0.18,
    "Logistics_Friction_Score": 0.16,
    "Demand_Volatility_Score": 0.12,
    "Concentration_Risk_Score": 0.12,
    "Historical_Instability_Score": 0.14,
    "Cost_Pressure_Score": 0.08,
}

FRIENDLY_NAMES = {
    "Geo_Weather_Exposure": "Geopolitical & Weather Exposure",
    "Supplier_Weakness_Score": "Supplier Weakness (reliability + financial health)",
    "Logistics_Friction_Score": "Logistics Friction (congestion, customs, lead time)",
    "Demand_Volatility_Score": "Demand Volatility & Low Inventory Buffer",
    "Concentration_Risk_Score": "Sourcing Concentration Risk",
    "Historical_Instability_Score": "Historical Disruption Instability",
    "Cost_Pressure_Score": "Cost & Fuel Price Pressure",
}

RECOMMENDATIONS = {
    "Geo_Weather_Exposure": [
        "Diversify sourcing across lower geopolitical-risk regions to reduce single-region exposure.",
        "Set up automated geopolitical & weather-alert monitoring for this lane.",
        "Pre-position safety stock ahead of known high-risk seasons (monsoon, hurricane, conflict cycles).",
    ],
    "Supplier_Weakness_Score": [
        "Initiate a supplier financial-health audit and request updated credit reports.",
        "Qualify at least one backup supplier for this category within 90 days.",
        "Tighten SLAs and introduce penalty/incentive clauses tied to on-time delivery.",
    ],
    "Logistics_Friction_Score": [
        "Evaluate an alternate transport mode or route to bypass congested ports.",
        "Pursue customs pre-clearance / trusted-trader programs to cut clearance delays.",
        "Rebook with buffer lead time during peak congestion windows.",
    ],
    "Demand_Volatility_Score": [
        "Improve demand forecasting cadence (move from monthly to weekly review).",
        "Raise safety-stock thresholds for this SKU/category to absorb variability.",
        "Introduce a sales & operations planning (S&OP) sync with key customers.",
    ],
    "Concentration_Risk_Score": [
        "Move from single-sourcing to a dual/multi-sourcing strategy for this category.",
        "Renegotiate contracts to include flexible volume/exit clauses.",
        "Build a qualified-supplier bench to shorten switching time in a crisis.",
    ],
    "Historical_Instability_Score": [
        "Place this supplier/lane on an active watchlist with monthly performance reviews.",
        "Explore supply-chain disruption insurance for high-instability lanes.",
        "Run a root-cause review of the last 3 disruption events with the supplier.",
    ],
    "Cost_Pressure_Score": [
        "Consider fuel-price hedging or fixed-rate freight contracts.",
        "Renegotiate pricing tiers or explore volume consolidation for better rates.",
        "Build a cost-contingency buffer into quarterly procurement budgets.",
    ],
}


def load_xgb_model():
    path = os.path.join(MODEL_DIR, "xgboost_model.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None


def load_artifacts(include_xgb=False):
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    target_encoder = joblib.load(os.path.join(MODEL_DIR, "target_encoder.pkl"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
    if include_xgb:
        xgb_model = load_xgb_model()
        return model, scaler, target_encoder, feature_cols, xgb_model
    return model, scaler, target_encoder, feature_cols



def compute_composite_scores(raw: dict) -> dict:
    """Compute the 7 explainable composite risk sub-scores (0-100) from raw inputs."""
    geo_weather = 0.55 * raw["Geopolitical_Risk_Index"] + 0.45 * raw["Weather_Risk_Index"]
    supplier_weak = 0.5 * (1 - raw["Supplier_Reliability_Score"]) + 0.5 * (1 - raw["Financial_Stability_Score"])
    logistics_friction = (
        0.4 * raw["Port_Congestion_Index"] +
        0.3 * np.clip(raw["Customs_Delay_Days"] / 10, 0, 1) +
        0.3 * np.clip(raw["Lead_Time_Days"] / 45, 0, 1)
    )
    demand_vol = 0.6 * (raw["Demand_Variability_Pct"] / 100) + 0.4 * (1 - np.clip(raw["Inventory_Level_Days"] / 30, 0, 1))
    concentration = 0.7 * raw["Single_Sourcing_Flag"] + 0.3 * (1 - raw["Contract_Flexibility_Score"])
    hist_instability = np.clip(
        (raw["Disruption_Events_Past_Year"] / 8) * 0.6 + (100 - raw["On_Time_Delivery_Rate_Pct"]) / 100 * 0.4, 0, 1)
    cost_pressure = np.clip((raw["Cost_Variance_Pct"] + 10) / 70 * 0.6 + raw["Fuel_Price_Index"] * 0.4, 0, 1)

    scores = {
        "Geo_Weather_Exposure": geo_weather * 100,
        "Supplier_Weakness_Score": supplier_weak * 100,
        "Logistics_Friction_Score": logistics_friction * 100,
        "Demand_Volatility_Score": demand_vol * 100,
        "Concentration_Risk_Score": concentration * 100,
        "Historical_Instability_Score": hist_instability * 100,
        "Cost_Pressure_Score": cost_pressure * 100,
    }
    return scores


def compute_ews(scores: dict, ref_min=8.0, ref_max=60.0) -> float:
    """Weighted Early Warning Score, scaled 0-100 using dataset-calibrated bounds."""
    raw_score = sum(scores[k] * w for k, w in EWS_WEIGHTS.items())
    scaled = ((raw_score - ref_min) / (ref_max - ref_min)) * 100
    return float(np.clip(scaled, 0, 100))


def ews_band(score: float) -> str:
    if score < 35:
        return "Safe"
    elif score < 60:
        return "Watch"
    elif score < 80:
        return "Elevated"
    return "Critical"


def explain(scores: dict, top_n=3):
    """Return top-N contributing risk factors ranked by weighted contribution."""
    contributions = {k: scores[k] * EWS_WEIGHTS[k] for k in scores}
    total = sum(contributions.values()) or 1
    ranked = sorted(contributions.items(), key=lambda x: -x[1])
    explanation = []
    for k, v in ranked[:top_n]:
        explanation.append({
            "factor": FRIENDLY_NAMES[k],
            "key": k,
            "raw_score": round(scores[k], 1),
            "contribution_pct": round(100 * v / total, 1),
        })
    return explanation


def recommend(explanation):
    recs = []
    for item in explanation:
        key = item["key"]
        recs.append({
            "factor": item["factor"],
            "actions": RECOMMENDATIONS[key],
        })
    return recs


def predict_ml(raw_input_encoded: dict, model, scaler, target_encoder, feature_cols):
    """raw_input_encoded must already include *_enc categorical columns."""
    X = pd.DataFrame([raw_input_encoded])[feature_cols]
    needs_scaling = model.__class__.__name__ in ["LogisticRegression", "KNeighborsClassifier", "SVC"]
    X_in = scaler.transform(X) if needs_scaling else X
    pred = model.predict(X_in)[0]
    proba = model.predict_proba(X_in)[0]
    label = target_encoder.inverse_transform([pred])[0]
    proba_dict = {cls: round(float(p), 4) for cls, p in zip(target_encoder.classes_, proba)}
    return label, proba_dict


REQUIRED_SHIPMENT_COLUMNS = [
    "Region", "Product_Category", "Supplier_Tier", "Transport_Mode",
    "Lead_Time_Days", "Distance_KM", "Order_Quantity", "Inventory_Level_Days",
    "Demand_Variability_Pct", "Supplier_Reliability_Score", "Financial_Stability_Score",
    "Geopolitical_Risk_Index", "Weather_Risk_Index", "Port_Congestion_Index",
    "Fuel_Price_Index", "Customs_Delay_Days", "Disruption_Events_Past_Year",
    "On_Time_Delivery_Rate_Pct", "Cost_Variance_Pct", "Single_Sourcing_Flag",
    "Contract_Flexibility_Score"
]


def validate_dataset_schema(df: pd.DataFrame) -> dict:
    """
    Validates an uploaded custom dataset for compatibility with SupplyGuard AI's
    preprocessing pipeline and XGBoost classification model.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return {
            "is_valid": False,
            "is_compatible": False,
            "error": "Dataset is not a valid DataFrame.",
            "present_columns": [],
            "missing_columns": REQUIRED_SHIPMENT_COLUMNS,
            "coverage_pct": 0.0,
            "warnings": [],
            "rows": 0,
            "columns_total": 0,
        }

    if df.empty or df.shape[1] == 0:
        return {
            "is_valid": False,
            "is_compatible": False,
            "error": "Uploaded dataset contains no data or 0 columns.",
            "present_columns": [],
            "missing_columns": REQUIRED_SHIPMENT_COLUMNS,
            "coverage_pct": 0.0,
            "warnings": [],
            "rows": len(df),
            "columns_total": df.shape[1],
        }

    cols = list(df.columns)
    present = [c for c in REQUIRED_SHIPMENT_COLUMNS if c in cols]
    missing = [c for c in REQUIRED_SHIPMENT_COLUMNS if c not in cols]
    coverage = round(100.0 * len(present) / len(REQUIRED_SHIPMENT_COLUMNS), 1)

    # Compatible if at least 50% of the domain features are present
    is_compatible = coverage >= 50.0

    warnings = []
    if missing and is_compatible:
        warnings.append(
            f"Missing {len(missing)} features ({', '.join(missing[:3])}{'...' if len(missing) > 3 else ''}). "
            "These will be defensively imputed with baseline domain medians during scoring."
        )

    null_counts = df[present].isna().sum() if present else pd.Series(dtype=int)
    null_cols = null_counts[null_counts > 0].to_dict()
    if null_cols:
        warnings.append(f"{len(null_cols)} columns contain missing values which will be cleaned during preprocessing.")

    return {
        "is_valid": True,
        "is_compatible": is_compatible,
        "present_columns": present,
        "missing_columns": missing,
        "coverage_pct": coverage,
        "warnings": warnings,
        "rows": len(df),
        "columns_total": len(cols),
    }


def batch_predict_df(df_input: pd.DataFrame, xgb_model, scaler, target_encoder, feature_cols, base_df=None):
    """
    Runs vectorized XGBoost disruption predictions, composite risk scoring,
    and Early Warning Scores across an entire dataset in batch.
    Never modifies df_input in-place.
    """
    df_scored = df_input.copy()
    n_rows = len(df_scored)
    if n_rows == 0:
        return df_scored, {"total": 0, "low": 0, "medium": 0, "high": 0, "high_pct": 0.0, "avg_risk_prob": 0.0, "avg_ews": 0.0}

    # Ensure Shipment_ID exists
    if "Shipment_ID" not in df_scored.columns:
        if "shipment_id" in df_scored.columns:
            df_scored["Shipment_ID"] = df_scored["shipment_id"]
        else:
            df_scored["Shipment_ID"] = [f"SG-UP-{i+1:05d}" for i in range(n_rows)]

    # Domain defaults if columns are missing
    defaults = {
        "Lead_Time_Days": 15.0,
        "Distance_KM": 4000.0,
        "Order_Quantity": 500.0,
        "Inventory_Level_Days": 15.0,
        "Demand_Variability_Pct": 20.0,
        "Supplier_Reliability_Score": 0.70,
        "Financial_Stability_Score": 0.65,
        "Geopolitical_Risk_Index": 0.30,
        "Weather_Risk_Index": 0.30,
        "Port_Congestion_Index": 0.30,
        "Fuel_Price_Index": 0.50,
        "Customs_Delay_Days": 2.0,
        "Disruption_Events_Past_Year": 1.0,
        "On_Time_Delivery_Rate_Pct": 85.0,
        "Cost_Variance_Pct": 10.0,
        "Single_Sourcing_Flag": 0,
        "Contract_Flexibility_Score": 0.50,
        "Region": "East Asia",
        "Product_Category": "Electronics",
        "Supplier_Tier": "Tier 1",
        "Transport_Mode": "Sea",
    }
    if base_df is not None:
        for k in defaults:
            if k in base_df.columns:
                if pd.api.types.is_numeric_dtype(base_df[k]):
                    defaults[k] = float(base_df[k].median())
                else:
                    defaults[k] = str(base_df[k].mode()[0])

    for k, default_val in defaults.items():
        if k not in df_scored.columns:
            df_scored[k] = default_val
        else:
            df_scored[k] = df_scored[k].fillna(default_val)

    # 1. Compute composite scores
    geo_weather = 0.55 * df_scored["Geopolitical_Risk_Index"] + 0.45 * df_scored["Weather_Risk_Index"]
    supplier_weak = 0.5 * (1 - df_scored["Supplier_Reliability_Score"]) + 0.5 * (1 - df_scored["Financial_Stability_Score"])
    logistics_friction = (
        0.4 * df_scored["Port_Congestion_Index"] +
        0.3 * np.clip(df_scored["Customs_Delay_Days"] / 10, 0, 1) +
        0.3 * np.clip(df_scored["Lead_Time_Days"] / 45, 0, 1)
    )
    demand_vol = 0.6 * (df_scored["Demand_Variability_Pct"] / 100) + 0.4 * (1 - np.clip(df_scored["Inventory_Level_Days"] / 30, 0, 1))
    concentration = 0.7 * df_scored["Single_Sourcing_Flag"] + 0.3 * (1 - df_scored["Contract_Flexibility_Score"])
    hist_instability = np.clip(
        (df_scored["Disruption_Events_Past_Year"] / 8) * 0.6 + (100 - df_scored["On_Time_Delivery_Rate_Pct"]) / 100 * 0.4, 0, 1)
    cost_pressure = np.clip((df_scored["Cost_Variance_Pct"] + 10) / 70 * 0.6 + df_scored["Fuel_Price_Index"] * 0.4, 0, 1)

    df_scored["Geo_Weather_Exposure"] = geo_weather * 100
    df_scored["Supplier_Weakness_Score"] = supplier_weak * 100
    df_scored["Logistics_Friction_Score"] = logistics_friction * 100
    df_scored["Demand_Volatility_Score"] = demand_vol * 100
    df_scored["Concentration_Risk_Score"] = concentration * 100
    df_scored["Historical_Instability_Score"] = hist_instability * 100
    df_scored["Cost_Pressure_Score"] = cost_pressure * 100

    # 2. Weighted Early Warning Score
    raw_ews = sum(df_scored[k] * w for k, w in EWS_WEIGHTS.items())
    ref_min, ref_max = 8.0, 60.0
    df_scored["Early_Warning_Score"] = np.clip(((raw_ews - ref_min) / (ref_max - ref_min)) * 100, 0, 100).round(1)
    df_scored["EWS_Band"] = df_scored["Early_Warning_Score"].apply(ews_band)

    # 3. Categorical encoding
    cat_cols = ["Region", "Product_Category", "Supplier_Tier", "Transport_Mode"]
    for col in cat_cols:
        enc_col = f"{col}_enc"
        if base_df is not None and col in base_df.columns:
            unique_cats = sorted(base_df[col].dropna().unique())
            mapping = {v: i for i, v in enumerate(unique_cats)}
        else:
            unique_cats = sorted(df_scored[col].dropna().unique())
            mapping = {v: i for i, v in enumerate(unique_cats)}
        df_scored[enc_col] = df_scored[col].map(lambda x: mapping.get(x, 0))

    # 4. Prepare feature matrix matching feature_cols
    X = pd.DataFrame(df_scored[feature_cols])

    # 5. Predict using xgb_model
    preds_raw = xgb_model.predict(X)
    preds_label = target_encoder.inverse_transform(preds_raw)
    df_scored["Predicted_Risk_Level"] = preds_label
    df_scored["xgboost_prediction"] = preds_label

    proba_mat = xgb_model.predict_proba(X)
    classes = list(target_encoder.classes_)
    for idx, c in enumerate(classes):
        df_scored[f"P_{c}"] = proba_mat[:, idx].round(4)

    high_idx = classes.index("High") if "High" in classes else 0
    df_scored["Disruption_Probability_High"] = proba_mat[:, high_idx].round(4)
    df_scored["risk_probability"] = df_scored["Disruption_Probability_High"]

    # 6. Identify top risk factor per row
    def get_top_factor(row):
        best_factor = "Geo_Weather_Exposure"
        max_contrib = -1.0
        for k, w in EWS_WEIGHTS.items():
            contrib = row.get(k, 0) * w
            if contrib > max_contrib:
                max_contrib = contrib
                best_factor = k
        return FRIENDLY_NAMES.get(best_factor, best_factor)

    df_scored["Top_Risk_Factor"] = df_scored.apply(get_top_factor, axis=1)

    low_cnt = int((df_scored["Predicted_Risk_Level"] == "Low").sum())
    med_cnt = int((df_scored["Predicted_Risk_Level"] == "Medium").sum())
    high_cnt = int((df_scored["Predicted_Risk_Level"] == "High").sum())

    summary = {
        "total": n_rows,
        "low": low_cnt,
        "medium": med_cnt,
        "high": high_cnt,
        "high_pct": round(100.0 * high_cnt / max(n_rows, 1), 1),
        "avg_risk_prob": round(float(df_scored["Disruption_Probability_High"].mean()), 4),
        "avg_ews": round(float(df_scored["Early_Warning_Score"].mean()), 1),
    }

    return df_scored, summary

