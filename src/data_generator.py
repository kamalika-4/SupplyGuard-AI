"""
SupplyGuard AI - Synthetic Dataset Generator
=============================================
Generates a realistic 5,000-row 'Global Supply Chain Risk & Logistics' dataset.
The data-generating process encodes real-world domain logic (weather, geopolitics,
supplier reliability, congestion, etc. driving disruption risk) so that downstream
EDA and ML actually find meaningful, explainable signal instead of pure noise.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 5000

regions = ["North America", "Europe", "East Asia", "South Asia", "Southeast Asia",
           "Middle East", "Africa", "South America", "Oceania"]

country_map = {
    "North America": ["USA", "Canada", "Mexico"],
    "Europe": ["Germany", "France", "Netherlands", "Poland", "Italy"],
    "East Asia": ["China", "Japan", "South Korea", "Taiwan"],
    "South Asia": ["India", "Bangladesh", "Pakistan"],
    "Southeast Asia": ["Vietnam", "Thailand", "Indonesia", "Malaysia"],
    "Middle East": ["UAE", "Saudi Arabia", "Turkey"],
    "Africa": ["South Africa", "Egypt", "Nigeria", "Kenya"],
    "South America": ["Brazil", "Chile", "Argentina"],
    "Oceania": ["Australia", "New Zealand"]
}

product_categories = ["Electronics", "Automotive Parts", "Pharmaceuticals", "Textiles",
                       "Food & Beverage", "Chemicals", "Machinery", "Consumer Goods",
                       "Raw Materials", "Semiconductors"]

transport_modes = ["Sea", "Air", "Road", "Rail", "Multimodal"]

supplier_tiers = ["Tier 1", "Tier 2", "Tier 3"]

rows = []
for i in range(1, N + 1):
    region = np.random.choice(regions, p=[0.14, 0.16, 0.18, 0.10, 0.12, 0.08, 0.08, 0.08, 0.06])
    country = np.random.choice(country_map[region])
    category = np.random.choice(product_categories)
    mode = np.random.choice(transport_modes, p=[0.35, 0.15, 0.25, 0.15, 0.10])
    tier = np.random.choice(supplier_tiers, p=[0.5, 0.35, 0.15])

    # --- Base risk drivers (domain-informed random variables) ---
    # Regions with higher baseline geopolitical & weather exposure
    geo_base = {"North America": 0.15, "Europe": 0.15, "East Asia": 0.35, "South Asia": 0.45,
                "Southeast Asia": 0.4, "Middle East": 0.55, "Africa": 0.5, "South America": 0.4,
                "Oceania": 0.1}[region]
    weather_base = {"North America": 0.3, "Europe": 0.25, "East Asia": 0.45, "South Asia": 0.55,
                     "Southeast Asia": 0.6, "Middle East": 0.35, "Africa": 0.45, "South America": 0.4,
                     "Oceania": 0.3}[region]

    geopolitical_risk_index = np.clip(np.random.normal(geo_base, 0.15), 0, 1)
    weather_risk_index = np.clip(np.random.normal(weather_base, 0.15), 0, 1)

    port_congestion_index = np.clip(np.random.normal(0.4 if mode in ["Sea", "Multimodal"] else 0.2, 0.15), 0, 1)
    fuel_price_index = np.clip(np.random.normal(0.5, 0.18), 0, 1)

    supplier_reliability_score = np.clip(np.random.normal(
        0.75 if tier == "Tier 1" else (0.6 if tier == "Tier 2" else 0.45), 0.15), 0.05, 1)

    financial_stability_score = np.clip(np.random.normal(
        0.7 if tier == "Tier 1" else (0.55 if tier == "Tier 2" else 0.4), 0.15), 0.05, 1)

    lead_time_days = max(1, int(np.random.normal(
        {"Air": 5, "Road": 8, "Rail": 12, "Sea": 30, "Multimodal": 20}[mode], 4)))

    distance_km = max(50, int(np.random.normal(
        {"Air": 6000, "Road": 1500, "Rail": 3000, "Sea": 9000, "Multimodal": 5000}[mode], 2000)))

    order_quantity = max(10, int(np.random.exponential(500)))
    inventory_level_days = max(0, np.random.normal(15, 8))
    demand_variability_pct = np.clip(np.random.normal(20, 12), 0, 100)

    customs_delay_days = max(0, np.random.exponential(2) * (1 + geopolitical_risk_index))
    disruption_events_past_year = np.random.poisson(1 + 3 * (geopolitical_risk_index + weather_risk_index) / 2)

    on_time_delivery_rate_pct = np.clip(
        100 - (disruption_events_past_year * 4) - (customs_delay_days * 1.5)
        + np.random.normal(supplier_reliability_score * 15, 8), 20, 100)

    cost_variance_pct = np.clip(
        (1 - supplier_reliability_score) * 15 + fuel_price_index * 10 + np.random.normal(0, 5), -10, 60)

    single_sourcing_flag = np.random.choice([1, 0], p=[0.3, 0.7])
    contract_flexibility_score = np.clip(np.random.normal(0.5, 0.2), 0, 1)

    # --- Composite latent disruption risk (drives target label) ---
    latent_risk = (
        0.20 * geopolitical_risk_index +
        0.18 * weather_risk_index +
        0.14 * port_congestion_index +
        0.12 * (1 - supplier_reliability_score) +
        0.10 * (1 - financial_stability_score) +
        0.08 * (disruption_events_past_year / 8) +
        0.06 * (customs_delay_days / 10) +
        0.05 * single_sourcing_flag +
        0.04 * (demand_variability_pct / 100) +
        0.03 * (1 - contract_flexibility_score)
    )
    latent_risk = np.clip(latent_risk + np.random.normal(0, 0.06), 0, 1)

    rows.append({
        "Shipment_ID": f"SG{i:05d}",
        "Region": region,
        "Country": country,
        "Product_Category": category,
        "Supplier_Tier": tier,
        "Transport_Mode": mode,
        "Lead_Time_Days": lead_time_days,
        "Distance_KM": distance_km,
        "Order_Quantity": order_quantity,
        "Inventory_Level_Days": round(inventory_level_days, 1),
        "Demand_Variability_Pct": round(demand_variability_pct, 1),
        "Supplier_Reliability_Score": round(supplier_reliability_score, 3),
        "Financial_Stability_Score": round(financial_stability_score, 3),
        "Geopolitical_Risk_Index": round(geopolitical_risk_index, 3),
        "Weather_Risk_Index": round(weather_risk_index, 3),
        "Port_Congestion_Index": round(port_congestion_index, 3),
        "Fuel_Price_Index": round(fuel_price_index, 3),
        "Customs_Delay_Days": round(customs_delay_days, 2),
        "Disruption_Events_Past_Year": disruption_events_past_year,
        "On_Time_Delivery_Rate_Pct": round(on_time_delivery_rate_pct, 1),
        "Cost_Variance_Pct": round(cost_variance_pct, 1),
        "Single_Sourcing_Flag": single_sourcing_flag,
        "Contract_Flexibility_Score": round(contract_flexibility_score, 3),
        "Latent_Risk_Score": round(latent_risk, 3),
    })

df = pd.DataFrame(rows)

# Quantile-based thresholds -> realistic, healthier class balance
q_low, q_high = df["Latent_Risk_Score"].quantile([0.45, 0.80])
def classify(x):
    if x < q_low:
        return "Low"
    elif x < q_high:
        return "Medium"
    return "High"
df["Risk_Level"] = df["Latent_Risk_Score"].apply(classify)

# Inject some realistic missingness & light noise (for preprocessing demo)
for col, frac in [("Financial_Stability_Score", 0.03), ("Customs_Delay_Days", 0.02),
                   ("Inventory_Level_Days", 0.02), ("Contract_Flexibility_Score", 0.025)]:
    idx = df.sample(frac=frac, random_state=1).index
    df.loc[idx, col] = np.nan

# A few duplicate rows to demonstrate dedup step in preprocessing
dupes = df.sample(8, random_state=2)
df = pd.concat([df, dupes], ignore_index=True)
df = df.sample(frac=1, random_state=3).reset_index(drop=True)  # shuffle

out_path = "/home/claude/SupplyGuard_AI/data/global_supply_chain_risk_logistics.csv"
df.to_csv(out_path, index=False)
print("Saved:", out_path, "| shape:", df.shape)
print(df["Risk_Level"].value_counts())
