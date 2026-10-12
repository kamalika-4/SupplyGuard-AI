# 🛰️ SupplyGuard AI — Supply Chain Disruption Risk Prediction & Early Warning System

An end-to-end, error-free analytics + ML dashboard for predicting and explaining supply chain
disruption risk, built on a 5,000-row **Global Supply Chain Risk & Logistics** dataset.

SupplyGuard AI follows a **Predict → Explain → Recommend** philosophy: every shipment gets an
ML-predicted risk level, a transparent 0–100 **Early Warning Score**, a plain-English breakdown of
*why* it's risky, and concrete preventive actions — not just a black-box score.

---

## ✨ What makes this different

| Feature | Description |
|---|---|
| 🛰️ **Early Warning Score (EWS)** | A fully transparent, weighted 0–100 composite score (Safe / Watch / Elevated / Critical bands) built from 7 explainable risk sub-scores — not a black box. |
| 🧩 **Explainable Risk Factors** | Every prediction shows its top 3 contributing risk drivers with exact % contribution to the score. |
| 📋 **Risk Ranking** | A live, filterable, color-graded leaderboard of the riskiest shipments/suppliers across your network. |
| 🧪 **What-If Simulator** | Drag sliders on any driver (geopolitical risk, sourcing strategy, inventory buffer, etc.) and watch the Early Warning Score respond instantly. |
| 💊 **Preventive Recommendations** | Each top risk factor maps to 3 concrete, actionable mitigation steps — procurement-ready, not generic advice. |
| 🧠 **Model Lab** | Side-by-side comparison of 7 ML classifiers with full metrics, confusion matrix, and feature importances. |
| 📊 **Preprocessing & EDA** | Raw → preprocessed → feature-engineered datasets shown side-by-side, with a before/after data-quality summary and the 6 core EDA charts as interactive Plotly visuals. |

---

## 📁 Project Structure

```
SupplyGuard_AI/
├── data/
│   ├── global_supply_chain_risk_logistics.csv   # Raw dataset (5,000 rows, 25 columns)
│   ├── preprocessed_data.csv                    # After preprocessing (canonical name used by the dashboard)
│   ├── clean_supply_chain_data.csv              # Same preprocessed data, kept for backward compatibility
│   └── feature_engineered_data.csv              # After feature engineering (EWS, composite scores)
├── src/
│   ├── data_generator.py       # Generates the synthetic-but-realistic dataset
│   ├── preprocessing.py        # Cleaning: dedup, imputation, validation — saves preprocessed_data.csv
│   ├── feature_engineering.py  # Composite risk scores + Early Warning Score
│   ├── eda.py                  # Static EDA chart generation (saved to reports/figures)
│   └── train_models.py         # Trains & compares 7 ML models, saves best model + artifacts
├── models/
│   ├── best_model.pkl, scaler.pkl, target_encoder.pkl, feature_cols.pkl
│   ├── model_comparison.json   # Full metrics, confusion matrix, feature importances
│   └── model_comparison.csv
├── dashboard/
│   ├── app.py                  # Main Streamlit dashboard (8 pages)
│   └── risk_engine.py          # Shared Predict → Explain → Recommend logic
├── notebooks/
│   └── SupplyGuard_AI_Analysis.ipynb   # Full, pre-executed pipeline walkthrough
├── reports/
│   └── figures/                # 6 saved EDA charts (PNG)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Regenerate the full pipeline from scratch
The repo already ships with the dataset and trained models, so this step is optional —
run it only if you want to regenerate everything from a fresh random seed:
```bash
cd src
python3 data_generator.py       # -> data/global_supply_chain_risk_logistics.csv
python3 preprocessing.py        # -> data/preprocessed_data.csv (+ data/clean_supply_chain_data.csv)
python3 feature_engineering.py  # -> data/feature_engineered_data.csv
python3 eda.py                  # -> reports/figures/*.png
python3 train_models.py         # -> models/*.pkl, model_comparison.json
```

### 3. Launch the dashboard
```bash
cd dashboard
streamlit run app.py
```
Then open the local URL Streamlit prints (typically `http://localhost:8501`).

### 4. Explore the notebook (optional)
```bash
jupyter notebook notebooks/SupplyGuard_AI_Analysis.ipynb
```
The notebook is already executed end-to-end with saved outputs, so you can read it without
re-running anything, or re-run it to regenerate everything interactively.

---

## 🧭 Dashboard Pages

1. **🏠 Mission Control** — network-wide KPIs, risk distribution, regional hotspots, and an interactive
   **Supply Chain Network Topology (NetworkX)** mapping Suppliers, Transit Modes, and Product Categories
   with betweenness centrality bottleneck detection.
2. **📊 Preprocessing & EDA** — the consolidated, single exploratory analysis page featuring 4 deep tabs:
   Data Pipeline Stages (raw, preprocessed, feature-engineered, and data-quality before/after stats),
   Core EDA Visualizations (6 interactive Plotly charts), Deep-Dive Risk Distributions (with multidimensional
   filters for Region, Category, and Transport Mode), and Filtered Dataset Explorer with CSV export.
3. **🧠 Model Lab** — comprehensive ML comparison benchmark across 7 classifiers with interactive head-to-head
   model comparison, an **XGBoost Classifier Deep Dive** (confusion matrix, classification report, gain importances),
   and **SHAP Explainable AI** visualizations (interactive beeswarm summary plot, mean |SHAP| class breakdown,
   and non-linear dependence analysis).
4. **🔮 Predict · Explain · Recommend** — score an existing shipment or custom parameters using either the
   Best Model or XGBoost Classifier; inspect the Early Warning Score gauge, domain risk drivers, an interactive
   **SHAP Waterfall Attribution Chart** showing exact feature pushes (+/- log-odds), and priority preventive recommendations.
5. **📋 Risk Ranking** — sortable, filterable, color-graded leaderboard of highest-exposure shipments with CSV export
   and an interactive **Shipment Network Dependency Trace (NetworkX)** tracing supplier, transit corridor, and co-dependent shipments.
6. **🧪 What-If Simulator** — adjust key risk drivers on a baseline shipment and see the Early Warning Score respond live.
7. **💡 Business Insights** — executive-level strategic insights enriched with NetworkX transit bottleneck analysis and
   XGBoost/SHAP non-linear risk findings.

---

## 🧪 Methodology Summary

- **Preprocessing**: duplicate removal, median/mode imputation for missing values, range validation
  on all bounded indices (0–1 scores, percentages, non-negative durations).
- **Feature Engineering**: 7 domain-driven composite risk sub-scores (Geo/Weather Exposure,
  Supplier Weakness, Logistics Friction, Demand Volatility, Concentration Risk, Historical
  Instability, Cost Pressure), combined into a transparent weighted **Early Warning Score**.
- **Modeling**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, KNN,
  and SVM (RBF) trained on an 80/20 stratified split; compared on Accuracy, Precision, Recall,
  weighted F1, and ROC-AUC (OvR). The best model (by F1) is persisted for the dashboard.
- **Explainability**: risk factor contributions are derived directly from the same transparent
  weighted formula that produces the Early Warning Score — no black-box post-hoc approximation
  needed, so what you see is exactly what drove the number.

**Note on model accuracy**: realistic supply-chain risk labels carry inherent noise (unexpected
disruptions, imperfect data), so the ~65–68% accuracy / ~0.85 ROC-AUC achieved here is intentionally
realistic rather than artificially inflated — this is a decision-support triage tool, not an oracle.

---

## 📊 Dataset Description

`data/global_supply_chain_risk_logistics.csv` — 5,000 shipment records, 25 columns:

| Column | Description |
|---|---|
| Shipment_ID | Unique shipment identifier |
| Region / Country | Origin region and country |
| Product_Category | 10 product categories (Electronics, Pharmaceuticals, etc.) |
| Supplier_Tier | Tier 1 / 2 / 3 |
| Transport_Mode | Sea / Air / Road / Rail / Multimodal |
| Lead_Time_Days, Distance_KM, Order_Quantity | Logistics parameters |
| Inventory_Level_Days, Demand_Variability_Pct | Demand-side signals |
| Supplier_Reliability_Score, Financial_Stability_Score | Supplier health (0–1) |
| Geopolitical_Risk_Index, Weather_Risk_Index, Port_Congestion_Index, Fuel_Price_Index | External risk indices (0–1) |
| Customs_Delay_Days, Disruption_Events_Past_Year, On_Time_Delivery_Rate_Pct, Cost_Variance_Pct | Historical performance |
| Single_Sourcing_Flag, Contract_Flexibility_Score | Sourcing strategy |
| Latent_Risk_Score, Risk_Level | Ground-truth risk (Low / Medium / High) |

Additional engineered columns (`feature_engineered_data.csv`) add the 7 composite risk scores,
`Early_Warning_Score`, `EWS_Band`, and label-encoded categorical columns for ML.

---

## 🔧 Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · SHAP · NetworkX · Streamlit · Plotly · Matplotlib · Seaborn · joblib

---

Built for supply chain risk analysts who want to move from *"what happened"* dashboards to
*"what's about to happen, why, and what to do about it."*
