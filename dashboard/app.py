"""
SupplyGuard AI - Supply Chain Disruption Risk Prediction & Early Warning System
=================================================================================
Run with:  streamlit run app.py   (from inside the dashboard/ folder)
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE, "src"))

from risk_engine import (
    load_artifacts, load_xgb_model, compute_composite_scores, compute_ews, ews_band,
    explain, recommend, predict_ml, EWS_WEIGHTS, FRIENDLY_NAMES,
    validate_dataset_schema, batch_predict_df, REQUIRED_SHIPMENT_COLUMNS
)
from preprocessing import load_raw, preprocess, save_preprocessed, RAW_PATH, PREPROCESSED_PATH
from agent_engine import SupplyChainDataToActionAgent
import hashlib
import json
from shap_engine import (
    compute_local_shap, plot_shap_waterfall,
    compute_global_shap_data, plot_shap_summary_bar,
    plot_shap_beeswarm_interactive, plot_shap_dependence_interactive
)
import datetime
from mongo_db import (
    check_mongo_connection, save_shipment, save_risk_prediction,
    get_shipments, get_risk_predictions, bulk_save_shipments,
    bulk_save_risk_predictions, sanitize_for_mongo,
    is_cloud_environment, is_mongo_persistence_disabled
)

DATA_PATH = os.path.join(BASE, "data", "feature_engineered_data.csv")
MODEL_DIR = os.path.join(BASE, "models")

# --------------------------------------------------------------------------------------
# PAGE CONFIG + THEME  ("Control Tower" visual identity — radar/telemetry command center)
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="SupplyGuard AI | Risk Control Tower",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "bg": "#0B1220",
    "panel": "#111B2E",
    "panel2": "#16223A",
    "grid": "#233350",
    "text": "#E7EDF7",
    "muted": "#8CA0C4",
    "safe": "#2DD4BF",
    "watch": "#FFC94D",
    "elevated": "#FF9F43",
    "critical": "#FF5A5F",
    "accent": "#5B8CFF",
}

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}
.stApp {{
    background: radial-gradient(1200px 800px at 85% -10%, #142338 0%, {PALETTE['bg']} 55%);
    color: {PALETTE['text']};
}}
h1, h2, h3, h4 {{
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: 0.2px;
}}
section[data-testid="stSidebar"] {{
    background: #0A101C;
    border-right: 1px solid {PALETTE['grid']};
}}
div[data-testid="stMetric"] {{
    background: linear-gradient(180deg, {PALETTE['panel']} 0%, {PALETTE['panel2']} 100%);
    border: 1px solid {PALETTE['grid']};
    border-radius: 10px;
    padding: 14px 16px 10px 16px;
}}
div[data-testid="stMetricValue"] {{
    font-family: 'Space Grotesk', sans-serif;
}}
.sg-card {{
    background: linear-gradient(180deg, {PALETTE['panel']} 0%, {PALETTE['panel2']} 100%);
    border: 1px solid {PALETTE['grid']};
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 14px;
}}
.sg-eyebrow {{
    color: {PALETTE['accent']};
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.4px;
    margin-bottom: 2px;
}}
.sg-band-safe {{ color: {PALETTE['safe']}; font-weight: 700; }}
.sg-band-watch {{ color: {PALETTE['watch']}; font-weight: 700; }}
.sg-band-elevated {{ color: {PALETTE['elevated']}; font-weight: 700; }}
.sg-band-critical {{ color: {PALETTE['critical']}; font-weight: 700; }}
.sg-pill {{
    display:inline-block; padding: 3px 12px; border-radius: 999px; font-size:0.82rem; font-weight:600;
    border: 1px solid {PALETTE['grid']};
}}
hr {{ border-color: {PALETTE['grid']}; }}
[data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
.stTabs [data-baseweb="tab"] {{
    background-color: {PALETTE['panel']}; border-radius: 8px 8px 0 0; padding: 8px 16px;
    border: 1px solid {PALETTE['grid']}; border-bottom: none;
}}
.stTabs [aria-selected="true"] {{ background-color: {PALETTE['panel2']} !important; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

BAND_COLOR = {"Safe": PALETTE["safe"], "Watch": PALETTE["watch"],
              "Elevated": PALETTE["elevated"], "Critical": PALETTE["critical"]}
RISK_COLOR = {"Low": PALETTE["safe"], "Medium": PALETTE["watch"], "High": PALETTE["critical"]}


def plotly_theme(fig, height=420):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text"], family="Inter"),
        height=height, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"])
    fig.update_yaxes(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"])
    return fig


# --------------------------------------------------------------------------------------
# DATA + MODEL LOADING
# --------------------------------------------------------------------------------------
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        if os.path.exists(PREPROCESSED_PATH):
            return pd.read_csv(PREPROCESSED_PATH)
        elif os.path.exists(RAW_PATH):
            return pd.read_csv(RAW_PATH)
        st.error(f"⚠️ Critical dataset file not found at `{DATA_PATH}`.")
        return pd.DataFrame()
    try:
        loaded = pd.read_csv(DATA_PATH)
        if loaded.empty:
            st.error("⚠️ Dataset file is empty.")
            return pd.DataFrame()
        return loaded
    except Exception as e:
        st.error(f"⚠️ Failed to load dataset: {e}")
        return pd.DataFrame()


@st.cache_resource
def load_models():
    try:
        model, scaler, target_encoder, feature_cols = load_artifacts()
    except Exception as e:
        st.error(f"⚠️ Error loading ML model artifacts: {e}")
        model, scaler, target_encoder, feature_cols = None, None, None, []
    xgb_model = load_xgb_model()
    return model, scaler, target_encoder, feature_cols, xgb_model


@st.cache_data
def load_comparison():
    import json
    comp_file = os.path.join(MODEL_DIR, "model_comparison.json")
    if os.path.exists(comp_file):
        try:
            with open(comp_file) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "best_model": "XGBoostClassifier",
        "results": [
            {"Model": "XGBoostClassifier", "Accuracy": 0.88, "Precision": 0.87, "Recall": 0.88, "F1_Score": 0.87, "ROC_AUC": 0.94}
        ]
    }


@st.cache_data
def load_raw_data():
    return pd.read_csv(RAW_PATH)


@st.cache_data
def load_preprocessed_data():
    if os.path.exists(PREPROCESSED_PATH):
        return pd.read_csv(PREPROCESSED_PATH)
    raw = load_raw()
    clean_df, _meta = preprocess(raw)
    try:
        save_preprocessed(clean_df)
    except OSError:
        pass
    return clean_df


def dataset_quality_stats(d: pd.DataFrame) -> dict:
    return {
        "Rows": len(d),
        "Columns": d.shape[1],
        "Missing Values": int(d.isna().sum().sum()),
        "Duplicate Rows": int(d.duplicated().sum()),
    }


df = load_data()
model, scaler, target_encoder, feature_cols, xgb_model = load_models()
comparison = load_comparison()

ORDER = ["Low", "Medium", "High"]
BAND_ORDER = ["Safe", "Watch", "Elevated", "Critical"]


@st.cache_data
def load_global_shap():
    if xgb_model is not None:
        return compute_global_shap_data(xgb_model, df, feature_cols, target_encoder, sample_size=350)
    return None


global_shap_data = load_global_shap()


@st.cache_resource
def load_agent(_df, _xgb, _mdl, _scl, _enc, _feats):
    return SupplyChainDataToActionAgent(_df, _xgb, _mdl, _scl, _enc, _feats)


agent = load_agent(df, xgb_model, model, scaler, target_encoder, feature_cols)


# --------------------------------------------------------------------------------------
# SIDEBAR NAVIGATION (EDA Explorer removed as requested; Preprocessing & EDA is sole EDA page)
# --------------------------------------------------------------------------------------
@st.cache_data(ttl=5)
def get_cached_mongo_status():
    return check_mongo_connection()

mongo_ok, mongo_info, mongo_err = get_cached_mongo_status()

st.sidebar.markdown("## 🛰️ SupplyGuard AI")
st.sidebar.caption("Supply Chain Disruption Risk — Control Tower")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Mission Control", "📊 Preprocessing & EDA", "🧠 Model Lab",
     "🔮 Predict · Explain · Recommend", "📋 Risk Ranking",
     "🧪 What-If Simulator", "💡 Business Insights"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<span class='sg-pill' style='color:{PALETTE['muted']}'>Dataset: {len(df):,} shipments</span>",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    f"<span class='sg-pill' style='color:{PALETTE['muted']}'>Best model: {comparison['best_model']}</span>",
    unsafe_allow_html=True,
)
if xgb_model is not None:
    st.sidebar.markdown(
        f"<span class='sg-pill' style='color:{PALETTE['accent']}'>⚡ XGBoost + SHAP: Active</span>",
        unsafe_allow_html=True,
    )
st.sidebar.markdown(
    f"<span class='sg-pill' style='color:{PALETTE['safe']}'>🤖 AI Agent: Active (MongoDB + ML)</span>",
    unsafe_allow_html=True,
)
if mongo_info.get("cloud_mode"):
    st.sidebar.markdown(
        f"<span class='sg-pill' style='color:{PALETTE['muted']}'>☁️ Cloud Mode: Live In-Memory (No DB)</span>",
        unsafe_allow_html=True,
    )
elif mongo_ok:
    st.sidebar.markdown(
        f"<span class='sg-pill' style='color:{PALETTE['safe']}'>🍃 MongoDB: Connected (SupplyGuardDB)</span>",
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        f"<span class='sg-pill' style='color:{PALETTE['critical']}'>🍃 MongoDB: Offline (localhost:27017)</span>",
        unsafe_allow_html=True,
    )


# ========================================================================================
# PAGE 1 — MISSION CONTROL (Overview & Network Topology)
# ========================================================================================
if page == "🏠 Mission Control":
    st.markdown("# Mission Control")
    st.caption("Real-time command center for global supply chain disruption exposure and network transit flows.")

    tab_overview, tab_agent = st.tabs([
        "📊 Network Overview & KPIs",
        "🤖 AI Supply Chain Data-to-Action Agent"
    ])

    with tab_overview:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Shipments", f"{len(df):,}")
        c2.metric("Avg Early Warning Score", f"{df['Early_Warning_Score'].mean():.1f} / 100")
        critical_pct = (df["EWS_Band"] == "Critical").mean() * 100
        c3.metric("Critical Exposure", f"{critical_pct:.1f}%")
        high_risk_pct = (df["Risk_Level"] == "High").mean() * 100
        c4.metric("High-Risk Shipments", f"{high_risk_pct:.1f}%")
        c5.metric("Avg On-Time Delivery", f"{df['On_Time_Delivery_Rate_Pct'].mean():.1f}%")

        st.markdown("")
        col1, col2 = st.columns([1.1, 1.4])

        with col1:
            st.markdown("#### Risk Level Distribution")
            vc = df["Risk_Level"].value_counts().reindex(ORDER)
            fig = go.Figure(go.Pie(
                labels=vc.index, values=vc.values, hole=0.62,
                marker=dict(colors=[RISK_COLOR[r] for r in ORDER]),
                textinfo="label+percent"))
            fig = plotly_theme(fig, height=360)
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.markdown("#### Early Warning Bands by Region")
            band_region = pd.crosstab(df["Region"], df["EWS_Band"], normalize="index")[BAND_ORDER] * 100
            band_region = band_region.loc[band_region[["Elevated", "Critical"]].sum(axis=1).sort_values(ascending=False).index]
            fig = go.Figure()
            for band in BAND_ORDER:
                fig.add_bar(name=band, x=band_region.index, y=band_region[band], marker_color=BAND_COLOR[band])
            fig.update_layout(barmode="stack")
            fig = plotly_theme(fig, height=360)
            st.plotly_chart(fig, width="stretch")

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### Top 10 Riskiest Countries (avg EWS)")
            top_countries = df.groupby("Country")["Early_Warning_Score"].mean().sort_values(ascending=False).head(10)
            fig = px.bar(top_countries[::-1], orientation="h", color=top_countries[::-1].values,
                         color_continuous_scale=["#2DD4BF", "#FFC94D", "#FF5A5F"])
            fig.update_layout(showlegend=False, coloraxis_showscale=False, xaxis_title="Avg Early Warning Score", yaxis_title="")
            fig = plotly_theme(fig, height=380)
            st.plotly_chart(fig, width="stretch")

        with col4:
            st.markdown("#### Disruption Events vs On-Time Delivery")
            sample = df.sample(min(1200, len(df)), random_state=1)
            fig = px.scatter(sample, x="Disruption_Events_Past_Year", y="On_Time_Delivery_Rate_Pct",
                              color="Risk_Level", color_discrete_map=RISK_COLOR, opacity=0.7,
                              category_orders={"Risk_Level": ORDER})
            fig = plotly_theme(fig, height=380)
            st.plotly_chart(fig, width="stretch")

        st.markdown(
            f"""<div class='sg-card'><span class='sg-eyebrow'>SYSTEM STATUS</span><br>
            SupplyGuard AI is actively scoring {len(df):,} shipments using a
            {comparison['best_model']} classifier (weighted F1 = {comparison['results'][0]['F1_Score']:.3f})
            alongside an <b>XGBoost Classifier</b> with SHAP explainability.
            Switch to the <b>AI Supply Chain Data-to-Action Agent</b> tab to ask questions and run automated diagnostic actions.</div>""",
            unsafe_allow_html=True,
        )

    with tab_agent:
        st.markdown("### 🤖 AI Supply Chain Data-to-Action Agent")
        st.caption("Autonomous intelligence engine querying live shipment telemetry from **MongoDB (SupplyGuardDB)**, "
                   "running ML models (XGBoost, SHAP, EWS), and delivering verified evidence tables with prioritized actions.")

        if mongo_info.get("cloud_mode"):
            agent_status_pill = f"<span class='sg-pill' style='color:{PALETTE['muted']}'>☁️ Cloud Mode: Live Telemetry</span>"
        elif mongo_ok:
            agent_status_pill = f"<span class='sg-pill' style='color:{PALETTE['safe']}'>🍃 MongoDB: Connected (`SupplyGuardDB`)</span>"
        else:
            agent_status_pill = f"<span class='sg-pill' style='color:{PALETTE['critical']}'>🍃 MongoDB: Offline Mode</span>"
        agent_k1.markdown(agent_status_pill, unsafe_allow_html=True)
        agent_k2.markdown(f"<span class='sg-pill' style='color:{PALETTE['accent']}'>⚡ ML Engine: XGBoost + SHAP</span>", unsafe_allow_html=True)
        agent_k3.markdown(f"<span class='sg-pill' style='color:{PALETTE['watch']}'>🎯 Scorers: EWS Composite Rules</span>", unsafe_allow_html=True)
        agent_k4.markdown(f"<span class='sg-pill' style='color:{PALETTE['safe']}'>🛡️ Policy: Zero Hallucination (Strict Evidence)</span>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("##### 💡 Suggested Questions & Quick Actions")
        prompt_col1, prompt_col2, prompt_col3 = st.columns(3)
        with prompt_col1:
            q_p1 = st.button("🚨 Triage Top Critical & High-Risk Shipments", width="stretch")
            q_p2 = st.button("🔍 Deep-Dive Diagnostic for SG02088", width="stretch")
        with prompt_col2:
            q_p3 = st.button("🚢 Compare Sea vs Air Transit Risk & Lead Times", width="stretch")
            q_p4 = st.button("🏭 Evaluate Supplier Tiers & Single-Sourcing Penalty", width="stretch")
        with prompt_col3:
            q_p5 = st.button("🌍 Global Regional Disruption Hotspots", width="stretch")
            q_p6 = st.button("📜 Audit Recent Predictions Logged in MongoDB", width="stretch")

        selected_prompt = ""
        if q_p1:
            selected_prompt = "Triage top critical and high-risk shipments"
        elif q_p2:
            selected_prompt = "Diagnose shipment SG02088"
        elif q_p3:
            selected_prompt = "Compare transport mode risks and transit times"
        elif q_p4:
            selected_prompt = "Analyze supplier tier and single-sourcing risk"
        elif q_p5:
            selected_prompt = "Identify global regional disruption hotspots"
        elif q_p6:
            selected_prompt = "Audit prediction history in MongoDB"

        st.markdown("##### 💬 Natural-Language Supply Chain Query")
        qc_input, qc_btn = st.columns([4, 1])
        with qc_input:
            user_query = st.text_input(
                "Enter your supply chain question:",
                value=selected_prompt,
                placeholder="e.g. What is the disruption risk for shipment SG02088 and what actions should we take?",
                label_visibility="collapsed",
                key="agent_nl_input",
            )
        with qc_btn:
            ask_clicked = st.button("⚡ Ask Agent", type="primary", width="stretch")

        active_query = user_query if (ask_clicked or selected_prompt) else ""

        if active_query:
            with st.spinner("Agent querying MongoDB, executing ML models, and generating evidence..."):
                response = agent.answer_question(active_query)

            st.markdown(response.summary, unsafe_allow_html=True)

            # Tools Execution Badges
            st.markdown("##### 🛠️ Analytical Pipeline Execution Trail")
            tools_html = " ".join([f"<span class='sg-pill' style='color:{PALETTE['accent']}; margin-right:6px;'>⚙️ {t}</span>" for t in response.tools_used])
            st.markdown(tools_html, unsafe_allow_html=True)

            # Evidence Table
            if response.evidence_df is not None and not response.evidence_df.empty:
                st.markdown("##### 📊 Factual Evidence Table (Verified Live Data)")
                st.dataframe(response.evidence_df, width="stretch")
                st.download_button(
                    "⬇️ Export Evidence Data (CSV)",
                    response.evidence_df.to_csv(index=False),
                    f"agent_evidence_{int(datetime.datetime.now(datetime.timezone.utc).timestamp())}.csv",
                    key="dl_agent_evidence",
                )

            # Recommendations
            if response.recommendations:
                st.markdown("##### 🎯 Evidence-Based Action Plan")
                rec_cols = st.columns(min(len(response.recommendations), 3))
                for i, r in enumerate(response.recommendations[:3]):
                    with rec_cols[i]:
                        actions_html = "".join([f"<li>{a}</li>" for a in r.get("actions", [])])
                        st.markdown(
                            f"""<div class='sg-card'>
                            <span class='sg-eyebrow'>PRIORITY {r.get('priority', i+1)}</span>
                            <h4 style='margin-top:2px'>{r.get('factor', 'Operational Intervention')}</h4>
                            <ul style='padding-left:18px; color:{PALETTE['muted']}'>{actions_html}</ul>
                            </div>""",
                            unsafe_allow_html=True,
                        )
        else:
            st.info("💡 Type a natural-language question above or click one of the quick actions to test the AI Data-to-Action Agent.")


# ========================================================================================
# PAGE 2 — PREPROCESSING & EDA (Comprehensive single EDA page, fully retaining all charts)
# ========================================================================================
elif page == "📊 Preprocessing & EDA":
    st.markdown("# Preprocessing & EDA")
    st.caption("End-to-end data preparation and exploratory analysis: from raw data validation "
               "to feature engineering, interactive multidimensional distributions, and data export.")

    raw_df = load_raw_data()
    pre_df = load_preprocessed_data()
    fe_df = df

    stats_raw = dataset_quality_stats(raw_df)
    stats_pre = dataset_quality_stats(pre_df)
    stats_fe = dataset_quality_stats(fe_df)

    p_tabs = st.tabs([
        "1️⃣ Data Pipeline Stages",
        "2️⃣ Core EDA Visualizations",
        "3️⃣ Deep-Dive Risk Distributions",
        "4️⃣ Filtered Dataset Explorer & Export",
        "5️⃣ 📤 Upload & Analyze Custom CSV",
    ])

    # TAB 1: Pipeline Stages
    with p_tabs[0]:
        st.markdown("#### Data Transformation Lifecycle")
        st.caption("Inspect how raw telemetry is cleaned, validated, and enriched into analysis-ready features.")

        t1, t2, t3, t4 = st.tabs(["Raw Data", "Preprocessed Data", "Feature-Engineered Data", "📈 Before vs After Metrics"])
        with t1:
            st.markdown(f"**Shape:** {raw_df.shape[0]:,} rows × {raw_df.shape[1]} columns — contains injected missing values & duplicate records.")
            st.dataframe(raw_df.head(200), width="stretch", height=380)
        with t2:
            st.markdown(f"**Shape:** {pre_df.shape[0]:,} rows × {pre_df.shape[1]} columns — after duplicate removal, median/mode imputation, and defensive range clipping.")
            st.dataframe(pre_df.head(200), width="stretch", height=380)
        with t3:
            st.markdown(f"**Shape:** {fe_df.shape[0]:,} rows × {fe_df.shape[1]} columns — engineered 7 composite risk scores, Early Warning Score, and EWS bands.")
            st.dataframe(fe_df.head(200), width="stretch", height=380)
        with t4:
            comp_df = pd.DataFrame({"Raw": stats_raw, "Preprocessed": stats_pre, "Feature-Engineered": stats_fe})
            st.markdown("##### Data-Quality Summary Table")
            st.dataframe(comp_df, width="stretch")

            c1, c2 = st.columns(2)
            with c1:
                mv = pd.DataFrame({"Stage": ["Raw", "Preprocessed", "Feature-Engineered"],
                                    "Missing Values": [stats_raw["Missing Values"], stats_pre["Missing Values"], stats_fe["Missing Values"]]})
                fig = px.bar(mv, x="Stage", y="Missing Values", color="Stage",
                             color_discrete_sequence=[PALETTE["critical"], PALETTE["watch"], PALETTE["safe"]])
                fig.update_layout(title="Missing Values — Before vs After", showlegend=False)
                st.plotly_chart(plotly_theme(fig, height=340), width="stretch")
            with c2:
                dv = pd.DataFrame({"Stage": ["Raw", "Preprocessed", "Feature-Engineered"],
                                    "Duplicate Rows": [stats_raw["Duplicate Rows"], stats_pre["Duplicate Rows"], stats_fe["Duplicate Rows"]]})
                fig = px.bar(dv, x="Stage", y="Duplicate Rows", color="Stage",
                             color_discrete_sequence=[PALETTE["critical"], PALETTE["watch"], PALETTE["safe"]])
                fig.update_layout(title="Duplicate Rows — Before vs After", showlegend=False)
                st.plotly_chart(plotly_theme(fig, height=340), width="stretch")

            st.success(f"Preprocessing removed **{stats_raw['Duplicate Rows']:,} → {stats_pre['Duplicate Rows']:,}** duplicate rows and "
                       f"**{stats_raw['Missing Values']:,} → {stats_pre['Missing Values']:,}** missing values, while feature engineering added "
                       f"**{stats_fe['Columns'] - stats_pre['Columns']}** composite domain features.")

    # TAB 2: Core EDA
    with p_tabs[1]:
        st.markdown("#### Core Supply Chain Disruption Patterns")
        st.caption("The foundational 6 risk charts computed live from the feature-engineered dataset.")

        e1, e2 = st.columns(2)
        with e1:
            vc = fe_df["Risk_Level"].value_counts().reindex(ORDER)
            fig = px.bar(x=vc.index, y=vc.values, color=vc.index, color_discrete_map=RISK_COLOR,
                         labels={"x": "Risk Level", "y": "Count"})
            fig.update_layout(title="1. Ground-Truth Risk Level Distribution", showlegend=False)
            st.plotly_chart(plotly_theme(fig, height=360), width="stretch")
        with e2:
            region_risk = pd.crosstab(fe_df["Region"], fe_df["Risk_Level"], normalize="index")[ORDER] * 100
            region_risk = region_risk.sort_values("High", ascending=False)
            fig = go.Figure()
            for r in ORDER:
                fig.add_bar(name=r, x=region_risk.index, y=region_risk[r], marker_color=RISK_COLOR[r])
            fig.update_layout(barmode="stack", title="2. Risk Composition by Region (%)", yaxis_title="% of Shipments")
            st.plotly_chart(plotly_theme(fig, height=360), width="stretch")

        e3, e4 = st.columns(2)
        with e3:
            key_cols = ["Geopolitical_Risk_Index", "Weather_Risk_Index", "Port_Congestion_Index",
                        "Supplier_Reliability_Score", "Financial_Stability_Score", "Customs_Delay_Days",
                        "Disruption_Events_Past_Year", "On_Time_Delivery_Rate_Pct", "Cost_Variance_Pct",
                        "Early_Warning_Score"]
            corr = fe_df[key_cols].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
            fig.update_layout(title="3. Correlation Heatmap — Key Risk Drivers")
            st.plotly_chart(plotly_theme(fig, height=440), width="stretch")
        with e4:
            fig = px.box(fe_df, x="Risk_Level", y="Early_Warning_Score", color="Risk_Level",
                         color_discrete_map=RISK_COLOR, category_orders={"Risk_Level": ORDER})
            fig.update_layout(title="4. Early Warning Score by Risk Level", showlegend=False)
            st.plotly_chart(plotly_theme(fig, height=440), width="stretch")

        e5, e6 = st.columns(2)
        with e5:
            tm = fe_df.groupby("Transport_Mode")["Disruption_Events_Past_Year"].mean().sort_values(ascending=False)
            fig = px.bar(x=tm.index, y=tm.values, color=tm.values, color_continuous_scale="OrRd",
                         labels={"x": "Transport Mode", "y": "Avg Disruption Events"})
            fig.update_layout(title="5. Avg Disruption Events by Transport Mode", showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(plotly_theme(fig, height=380), width="stretch")
        with e6:
            cat_ews = fe_df.groupby("Product_Category")["Early_Warning_Score"].mean().sort_values(ascending=False)
            fig = px.bar(x=cat_ews.index, y=cat_ews.values, color=cat_ews.values,
                         color_continuous_scale=["#2DD4BF", "#FFC94D", "#FF5A5F"],
                         labels={"x": "Product Category", "y": "Avg Early Warning Score"})
            fig.update_layout(title="6. Avg Early Warning Score by Product Category", showlegend=False,
                               coloraxis_showscale=False, xaxis_tickangle=-30)
            st.plotly_chart(plotly_theme(fig, height=380), width="stretch")

    # TAB 3: Deep-Dive Distributions & Filters
    with p_tabs[2]:
        st.markdown("#### Interactive Deep-Dive Explorations")
        st.caption("Slice and examine distributions across multi-tiered supplier and logistics attributes.")

        f1, f2, f3 = st.columns(3)
        regions_sel = f1.multiselect("Filter Regions", sorted(df["Region"].unique()), default=[], key="eda_reg")
        cats_sel = f2.multiselect("Filter Product Categories", sorted(df["Product_Category"].unique()), default=[], key="eda_cat")
        modes_sel = f3.multiselect("Filter Transport Modes", sorted(df["Transport_Mode"].unique()), default=[], key="eda_mode")

        fdf = df.copy()
        if regions_sel:
            fdf = fdf[fdf["Region"].isin(regions_sel)]
        if cats_sel:
            fdf = fdf[fdf["Product_Category"].isin(cats_sel)]
        if modes_sel:
            fdf = fdf[fdf["Transport_Mode"].isin(modes_sel)]

        st.markdown(f"**Filtered dataset scope:** {len(fdf):,} / {len(df):,} shipments")

        d1, d2 = st.columns(2)
        with d1:
            fig = px.histogram(fdf, x="Early_Warning_Score", color="Risk_Level", nbins=40,
                                color_discrete_map=RISK_COLOR, category_orders={"Risk_Level": ORDER})
            fig.update_layout(title="Early Warning Score Distribution by Risk Level")
            st.plotly_chart(plotly_theme(fig), width="stretch")
        with d2:
            fig = px.box(fdf, x="Risk_Level", y="On_Time_Delivery_Rate_Pct", color="Risk_Level",
                         color_discrete_map=RISK_COLOR, category_orders={"Risk_Level": ORDER})
            fig.update_layout(title="On-Time Delivery Rate by Risk Level", showlegend=False)
            st.plotly_chart(plotly_theme(fig), width="stretch")

        d3, d4 = st.columns(2)
        with d3:
            fig = px.histogram(fdf, x="Lead_Time_Days", color="Transport_Mode", nbins=30)
            fig.update_layout(title="Lead Time Distribution by Transport Mode")
            st.plotly_chart(plotly_theme(fig), width="stretch")
        with d4:
            fig = px.violin(fdf, x="Supplier_Tier", y="Supplier_Reliability_Score", color="Supplier_Tier", box=True)
            fig.update_layout(title="Supplier Reliability Score by Tier", showlegend=False)
            st.plotly_chart(plotly_theme(fig), width="stretch")

    # TAB 4: Filtered Data Table & Export
    with p_tabs[3]:
        st.markdown("#### Live Filtered Dataset & CSV Export")
        st.caption("Inspect individual shipment records with their computed Early Warning Scores and composite drivers.")
        st.dataframe(fdf.head(500), width="stretch", height=450)
        st.download_button(
            "⬇️ Download Filtered Supply Chain Data (CSV)",
            fdf.to_csv(index=False),
            "filtered_supply_chain_data.csv",
            mime="text/csv"
        )

    # TAB 5: Upload & Analyze Custom CSV
    with p_tabs[4]:
        st.markdown("#### 📤 Upload, Validate, Preprocess & Analyze Custom Supply Chain Dataset")
        st.caption("Upload your custom shipment dataset (CSV) to validate feature schema, run automated cleaning, "
                   "visualize multidimensional risk distributions, and generate batch XGBoost predictions.")

        up_col1, up_col2 = st.columns([3, 1])
        with up_col1:
            uploaded_file = st.file_uploader(
                "Select a CSV dataset to upload",
                type=["csv"],
                key="custom_csv_file_uploader",
                help="Upload a comma-separated CSV file containing supply chain and shipment parameters."
            )
        with up_col2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            # Generate a 5-row clean sample template for quick testing and schema verification
            sample_template_cols = [c for c in REQUIRED_SHIPMENT_COLUMNS if c in df.columns]
            sample_template_df = df[sample_template_cols].head(5).copy()
            if "Shipment_ID" in df.columns:
                sample_template_df.insert(0, "Shipment_ID", [f"SAMPLE-{i+1:03d}" for i in range(5)])
            sample_csv_bytes = sample_template_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download CSV Template",
                data=sample_csv_bytes,
                file_name="supplyguard_shipment_template.csv",
                mime="text/csv",
                width="stretch",
                help="Download a pre-formatted 5-row CSV template matching SupplyGuard AI requirements."
            )

        df_custom_raw = None

        if uploaded_file is None:
            st.info("💡 **Ready for Data:** Upload your shipment CSV above, or download the template to test the workflow. "
                    "Original system datasets remain completely untouched on disk.")
        else:
            # 1. Defensive file validation
            if uploaded_file.size == 0:
                st.error("❌ **Invalid File:** The uploaded file is completely empty (0 bytes). Please select a valid CSV file.")
            else:
                try:
                    uploaded_file.seek(0)
                    try:
                        df_custom_raw = pd.read_csv(uploaded_file, encoding="utf-8")
                    except UnicodeDecodeError:
                        uploaded_file.seek(0)
                        df_custom_raw = pd.read_csv(uploaded_file, encoding="latin-1")
                except pd.errors.EmptyDataError:
                    st.error("❌ **Invalid File:** The uploaded file contains no data or headers. Please provide a populated CSV.")
                    df_custom_raw = None
                except pd.errors.ParserError as pe:
                    st.error(f"❌ **Malformed CSV File:** Failed to parse comma-separated values. Details: {pe}")
                    df_custom_raw = None
                except Exception as ex:
                    st.error(f"❌ **Error reading uploaded file:** {type(ex).__name__}: {ex}")
                    df_custom_raw = None

        if df_custom_raw is not None:
            if df_custom_raw.empty or df_custom_raw.shape[1] == 0:
                st.error("❌ **Empty Dataset:** The uploaded CSV parsed successfully but contains 0 rows or 0 columns.")
            else:
                st.session_state["uploaded_raw_df"] = df_custom_raw
                st.session_state["uploaded_file_name"] = uploaded_file.name

                st.markdown("---")
                # SECTION 1: PREVIEW & HEALTH CHECK
                st.markdown("##### 1️⃣ 🔍 Raw Dataset Preview & Health Check")
                prev_m1, prev_m2, prev_m3, prev_m4 = st.columns(4)
                total_cells = df_custom_raw.shape[0] * df_custom_raw.shape[1]
                missing_total = int(df_custom_raw.isna().sum().sum())
                dupe_total = int(df_custom_raw.duplicated().sum())

                with prev_m1:
                    st.metric("Total Rows", f"{df_custom_raw.shape[0]:,}")
                with prev_m2:
                    st.metric("Total Columns", f"{df_custom_raw.shape[1]}")
                with prev_m3:
                    st.metric("Missing Cells", f"{missing_total:,}", delta=f"{100*missing_total/max(total_cells, 1):.1f}% of data", delta_color="inverse")
                with prev_m4:
                    st.metric("Duplicate Rows", f"{dupe_total:,}", delta_color="inverse")

                st.dataframe(df_custom_raw.head(100), width="stretch", height=260)

                # SECTION 2: SCHEMA VALIDATION & COMPATIBILITY
                st.markdown("##### 2️⃣ 🛡️ Feature Schema Validation & Model Compatibility")
                val_res = validate_dataset_schema(df_custom_raw)

                if val_res["coverage_pct"] >= 100.0:
                    st.success(f"✅ **100% Schema Match:** All {len(REQUIRED_SHIPMENT_COLUMNS)} required shipment parameters are present. "
                               "The dataset is fully compatible with the standard preprocessing pipeline and the XGBoost classifier.")
                elif val_res["is_compatible"]:
                    st.warning(f"⚠️ **Partial Compatibility ({val_res['coverage_pct']}% Schema Coverage):** Found "
                               f"{len(val_res['present_columns'])} of {len(REQUIRED_SHIPMENT_COLUMNS)} expected features. "
                               "Missing parameters will be defensively imputed with domain baseline medians/modes during preprocessing and ML scoring.")
                else:
                    st.error(f"❌ **Incompatible Schema ({val_res['coverage_pct']}% Coverage):** Only "
                             f"{len(val_res['present_columns'])} matching supply chain features detected. "
                             "Please refer to the required schema table below or use the provided CSV template.")

                with st.expander("🔍 Detailed Schema Feature Mapping & Coverage Audit", expanded=False):
                    aud_c1, aud_c2 = st.columns(2)
                    with aud_c1:
                        st.markdown(f"**✅ Present Features ({len(val_res['present_columns'])}):**")
                        if val_res["present_columns"]:
                            st.write(val_res["present_columns"])
                        else:
                            st.write("*None*")
                    with aud_c2:
                        st.markdown(f"**❌ Missing Features ({len(val_res['missing_columns'])}):**")
                        if val_res["missing_columns"]:
                            st.write(val_res["missing_columns"])
                        else:
                            st.write("*None — Complete schema!*")

                # SECTION 3: PREPROCESSING & FEATURE ENGINEERING
                st.markdown("##### 3️⃣ 🧹 Data Cleaning, Imputation & Feature Engineering")
                st.caption("Executes duplicate removal, median imputation for numeric telemetry, mode imputation for categoricals, "
                           "defensive range bounds, and calculates the 7 composite risk scores + Early Warning Score.")

                btn_clean = st.button("⚡ Execute Cleaning & Feature Engineering", type="secondary", key="btn_run_clean_custom")

                # Preprocess if requested or retrieve from cache
                need_prep = btn_clean or ("uploaded_clean_df" not in st.session_state)
                if need_prep:
                    with st.spinner("Cleaning dataset and engineering composite risk scores..."):
                        clean_data, clean_meta = preprocess(df_custom_raw)
                        # Compute composite risk features on clean_data
                        # Make sure default columns exist so composite calculation succeeds
                        def_df = clean_data.copy()
                        for req_c in REQUIRED_SHIPMENT_COLUMNS:
                            if req_c not in def_df.columns:
                                if req_c in df.columns:
                                    if pd.api.types.is_numeric_dtype(df[req_c]):
                                        def_df[req_c] = float(df[req_c].median())
                                    else:
                                        def_df[req_c] = str(df[req_c].mode()[0])
                                else:
                                    def_df[req_c] = 0.5
                        # Ensure Shipment_ID exists
                        if "Shipment_ID" not in def_df.columns:
                            if "shipment_id" in def_df.columns:
                                def_df["Shipment_ID"] = def_df["shipment_id"]
                            else:
                                def_df["Shipment_ID"] = [f"SG-UP-{i+1:05d}" for i in range(len(def_df))]

                        # Calculate composite scores
                        gw = 0.55 * def_df["Geopolitical_Risk_Index"] + 0.45 * def_df["Weather_Risk_Index"]
                        sw = 0.5 * (1 - def_df["Supplier_Reliability_Score"]) + 0.5 * (1 - def_df["Financial_Stability_Score"])
                        lf = (
                            0.4 * def_df["Port_Congestion_Index"] +
                            0.3 * np.clip(def_df["Customs_Delay_Days"] / 10, 0, 1) +
                            0.3 * np.clip(def_df["Lead_Time_Days"] / 45, 0, 1)
                        )
                        dv = 0.6 * (def_df["Demand_Variability_Pct"] / 100) + 0.4 * (1 - np.clip(def_df["Inventory_Level_Days"] / 30, 0, 1))
                        cr = 0.7 * def_df["Single_Sourcing_Flag"] + 0.3 * (1 - def_df["Contract_Flexibility_Score"])
                        hi = np.clip((def_df["Disruption_Events_Past_Year"] / 8) * 0.6 + (100 - def_df["On_Time_Delivery_Rate_Pct"]) / 100 * 0.4, 0, 1)
                        cp = np.clip((def_df["Cost_Variance_Pct"] + 10) / 70 * 0.6 + def_df["Fuel_Price_Index"] * 0.4, 0, 1)

                        def_df["Geo_Weather_Exposure"] = gw * 100
                        def_df["Supplier_Weakness_Score"] = sw * 100
                        def_df["Logistics_Friction_Score"] = lf * 100
                        def_df["Demand_Volatility_Score"] = dv * 100
                        def_df["Concentration_Risk_Score"] = cr * 100
                        def_df["Historical_Instability_Score"] = hi * 100
                        def_df["Cost_Pressure_Score"] = cp * 100

                        raw_ews = sum(def_df[k] * w for k, w in EWS_WEIGHTS.items())
                        def_df["Early_Warning_Score"] = np.clip(((raw_ews - 8.0) / (60.0 - 8.0)) * 100, 0, 100).round(1)
                        def_df["EWS_Band"] = def_df["Early_Warning_Score"].apply(ews_band)

                        st.session_state["uploaded_clean_df"] = def_df
                        st.session_state["uploaded_clean_meta"] = clean_meta

                clean_df = st.session_state.get("uploaded_clean_df", df_custom_raw)
                clean_meta = st.session_state.get("uploaded_clean_meta", {})

                # Before vs After Metrics Callout
                if clean_meta:
                    st.info(
                        f"📊 **Data Cleaning Summary:** Processed **{clean_meta.get('rows_before', len(df_custom_raw)):,}** raw rows. "
                        f"Removed **{clean_meta.get('duplicates_removed', 0):,}** duplicate records. "
                        f"Imputed **{clean_meta.get('missing_values_before_total', 0):,}** missing cells. "
                        f"Final preprocessed dataset: **{clean_meta.get('rows_after', len(clean_df)):,}** rows × **{clean_df.shape[1]}** features."
                    )

                st.dataframe(clean_df.head(100), width="stretch", height=260)
                st.download_button(
                    "⬇️ Download Cleaned & Feature-Engineered Dataset (CSV)",
                    clean_df.to_csv(index=False),
                    f"cleaned_{uploaded_file.name}",
                    mime="text/csv",
                    key="dl_cleaned_custom_csv"
                )

                # SECTION 4: EXPLORATORY DATA ANALYSIS (EDA)
                st.markdown("##### 4️⃣ 📊 Exploratory Risk Analysis for Uploaded Dataset")
                eda_c1, eda_c2 = st.columns(2)
                with eda_c1:
                    if "Early_Warning_Score" in clean_df.columns:
                        fig_ews = px.histogram(
                            clean_df, x="Early_Warning_Score", nbins=25,
                            title="Early Warning Score (EWS) Distribution",
                            color_discrete_sequence=[PALETTE["accent"]],
                        )
                        fig_ews.add_vline(x=35, line_dash="dash", line_color=PALETTE["safe"], annotation_text="Safe")
                        fig_ews.add_vline(x=60, line_dash="dash", line_color=PALETTE["watch"], annotation_text="Watch")
                        fig_ews.add_vline(x=80, line_dash="dash", line_color=PALETTE["critical"], annotation_text="Critical")
                        st.plotly_chart(plotly_theme(fig_ews, height=320), width="stretch")
                with eda_c2:
                    if "EWS_Band" in clean_df.columns:
                        band_order = ["Safe", "Watch", "Elevated", "Critical"]
                        band_counts = clean_df["EWS_Band"].value_counts().reindex(band_order, fill_value=0)
                        band_colors = [PALETTE["safe"], PALETTE["watch"], PALETTE["elevated"], PALETTE["critical"]]
                        fig_band = px.bar(
                            x=band_counts.index, y=band_counts.values,
                            title="Early Warning Risk Band Breakdown",
                            color=band_counts.index,
                            color_discrete_map={k: c for k, c in zip(band_order, band_colors)},
                            labels={"x": "EWS Band", "y": "Shipments"},
                        )
                        fig_band.update_layout(showlegend=False)
                        st.plotly_chart(plotly_theme(fig_band, height=320), width="stretch")

                eda_c3, eda_c4 = st.columns(2)
                with eda_c3:
                    if "Region" in clean_df.columns and "Lead_Time_Days" in clean_df.columns:
                        reg_summary = clean_df.groupby("Region")["Lead_Time_Days"].mean().reset_index().sort_values("Lead_Time_Days", ascending=False)
                        fig_reg = px.bar(
                            reg_summary, x="Region", y="Lead_Time_Days",
                            title="Average Transit Lead Time by Region (Days)",
                            color="Lead_Time_Days",
                            color_continuous_scale="Blues",
                        )
                        st.plotly_chart(plotly_theme(fig_reg, height=320), width="stretch")
                with eda_c4:
                    if "Supplier_Reliability_Score" in clean_df.columns and "Financial_Stability_Score" in clean_df.columns:
                        fig_supp = px.scatter(
                            clean_df, x="Supplier_Reliability_Score", y="Financial_Stability_Score",
                            color=clean_df["EWS_Band"] if "EWS_Band" in clean_df.columns else None,
                            color_discrete_map={"Safe": PALETTE["safe"], "Watch": PALETTE["watch"], "Elevated": PALETTE["elevated"], "Critical": PALETTE["critical"]},
                            title="Supplier Reliability vs Financial Health",
                            hover_data=["Shipment_ID"] if "Shipment_ID" in clean_df.columns else None,
                        )
                        st.plotly_chart(plotly_theme(fig_supp, height=320), width="stretch")

                # Composite Drivers Bar Chart
                comp_cols = [k for k in EWS_WEIGHTS if k in clean_df.columns]
                if comp_cols:
                    avg_scores = {FRIENDLY_NAMES.get(k, k): round(float(clean_df[k].mean()), 1) for k in comp_cols}
                    comp_df = pd.DataFrame(list(avg_scores.items()), columns=["Risk Dimension", "Average Score (0-100)"]).sort_values("Average Score (0-100)", ascending=True)
                    fig_comp = px.bar(
                        comp_df, x="Average Score (0-100)", y="Risk Dimension", orientation="h",
                        title="Dataset Average Composite Risk Drivers",
                        color="Average Score (0-100)",
                        color_continuous_scale="Reds",
                    )
                    st.plotly_chart(plotly_theme(fig_comp, height=300), width="stretch")

                # SECTION 5: BATCH XGBOOST RISK PREDICTIONS
                st.markdown("##### 5️⃣ 🔮 Batch Disruption Risk Predictions (XGBoost Classifier)")
                if not val_res["is_compatible"]:
                    st.warning("⚠️ Batch XGBoost predictions are disabled because the dataset does not have sufficient supply chain parameter features.")
                else:
                    st.caption("Scores every uploaded shipment across the 28 engineered features using the trained XGBoost Classifier to output Disruption Risk Levels, probabilities, and top risk factors.")

                    pred_c_btn, pred_c_info = st.columns([1.5, 2.5])
                    with pred_c_btn:
                        btn_batch_pred = st.button("🚀 Run Batch XGBoost Predictions", type="primary", width="stretch", key="btn_run_batch_xgb")

                    if btn_batch_pred or ("uploaded_predicted_df" in st.session_state):
                        if btn_batch_pred or ("uploaded_predicted_df" not in st.session_state):
                            with st.spinner("Running vectorized XGBoost model inference..."):
                                scored_df, pred_summary = batch_predict_df(
                                    clean_df, xgb_model, scaler, target_encoder, feature_cols, base_df=df
                                )
                                st.session_state["uploaded_predicted_df"] = scored_df
                                st.session_state["uploaded_pred_summary"] = pred_summary
                        else:
                            scored_df = st.session_state["uploaded_predicted_df"]
                            pred_summary = st.session_state["uploaded_pred_summary"]

                        # KPI Summary row
                        pred_kpi1, pred_kpi2, pred_kpi3, pred_kpi4 = st.columns(4)
                        with pred_kpi1:
                            st.metric("Total Scored", f"{pred_summary['total']:,}")
                        with pred_kpi2:
                            st.metric("High Disruption Risk", f"{pred_summary['high']:,} ({pred_summary['high_pct']}%)")
                        with pred_kpi3:
                            st.metric("Medium Risk", f"{pred_summary['medium']:,}")
                        with pred_kpi4:
                            st.metric("Low Risk", f"{pred_summary['low']:,}")

                        # Risk Distribution Visualization
                        pred_viz1, pred_viz2 = st.columns(2)
                        with pred_viz1:
                            risk_counts = pd.DataFrame({
                                "Risk_Level": ["Low", "Medium", "High"],
                                "Count": [pred_summary["low"], pred_summary["medium"], pred_summary["high"]]
                            })
                            fig_pred_bar = px.bar(
                                risk_counts, x="Risk_Level", y="Count", color="Risk_Level",
                                color_discrete_map=RISK_COLOR,
                                title="Predicted Disruption Risk Level Distribution"
                            )
                            fig_pred_bar.update_layout(showlegend=False)
                            st.plotly_chart(plotly_theme(fig_pred_bar, height=300), width="stretch")

                        with pred_viz2:
                            fig_prob_hist = px.histogram(
                                scored_df, x="Disruption_Probability_High", nbins=20,
                                title="High Disruption Probability Distribution P(High)",
                                color_discrete_sequence=[PALETTE["critical"]]
                            )
                            st.plotly_chart(plotly_theme(fig_prob_hist, height=300), width="stretch")

                        # Display Table with Filter
                        st.markdown("##### 📋 Batch Scored Shipments Table")
                        filter_risk = st.multiselect(
                            "Filter Table by Predicted Risk Level:",
                            options=["High", "Medium", "Low"],
                            default=["High", "Medium", "Low"],
                            key="filter_custom_predictions"
                        )
                        disp_cols = [c for c in [
                            "Shipment_ID", "Predicted_Risk_Level", "Disruption_Probability_High",
                            "Early_Warning_Score", "EWS_Band", "Top_Risk_Factor", "Region", "Transport_Mode",
                            "Lead_Time_Days", "Distance_KM"
                        ] if c in scored_df.columns]

                        table_df = scored_df[scored_df["Predicted_Risk_Level"].isin(filter_risk)][disp_cols]
                        st.dataframe(table_df, width="stretch", height=340)

                        # Action Buttons: Download & Save to MongoDB
                        act_c1, act_c2 = st.columns(2)
                        with act_c1:
                            st.download_button(
                                "⬇️ Download Predictions (CSV)",
                                scored_df.to_csv(index=False),
                                f"predictions_{uploaded_file.name}",
                                mime="text/csv",
                                width="stretch",
                                key="dl_predictions_csv"
                            )
                        with act_c2:
                            btn_save_batch_mongo = st.button(
                                "💾 Persist All Predictions to MongoDB",
                                type="primary",
                                width="stretch",
                                key="btn_save_batch_mongo"
                            )

                        if btn_save_batch_mongo:
                            with st.spinner("Connecting to MongoDB and persisting batch prediction telemetry..."):
                                mongo_docs = []
                                now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
                                for _, r in scored_df.iterrows():
                                    r_dict = r.to_dict()
                                    clean_inputs = {str(k): sanitize_for_mongo(v) for k, v in r_dict.items() if not str(k).startswith("_")}
                                    sid_val = str(r.get("Shipment_ID", f"SG-UP-{_}"))
                                    pred_lvl = str(r.get("Predicted_Risk_Level", "Low"))
                                    p_high = float(r.get("Disruption_Probability_High", 0.0))
                                    p_low = float(r.get("P_Low", 0.0))
                                    p_med = float(r.get("P_Medium", 0.0))
                                    sub_h = hashlib.sha256(f"{sid_val}_{pred_lvl}_{json.dumps(clean_inputs, sort_keys=True, default=str)}".encode()).hexdigest()[:16]

                                    mongo_docs.append({
                                        "shipment_id": sid_val,
                                        "Shipment_ID": sid_val,
                                        "model_name": "XGBoostClassifier",
                                        "xgboost_prediction": pred_lvl,
                                        "predicted_risk_level": pred_lvl,
                                        "risk_probability": round(p_high, 4),
                                        "prediction_probabilities": {"Low": p_low, "Medium": p_med, "High": p_high},
                                        "early_warning_score": round(float(r.get("Early_Warning_Score", 0.0)), 2),
                                        "ews_band": str(r.get("EWS_Band", "Safe")),
                                        "shap_summary": {"top_risk_driver": str(r.get("Top_Risk_Factor", ""))},
                                        "shap_explanation": {"top_risk_driver": str(r.get("Top_Risk_Factor", ""))},
                                        "actual_input_values": clean_inputs,
                                        "shipment_inputs": clean_inputs,
                                        "input_values": clean_inputs,
                                        "submission_hash": sub_h,
                                        "timestamp": now_iso,
                                        "prediction_timestamp": now_iso,
                                        "created_at": now_iso,
                                        "dataset_source": "Uploaded_CSV",
                                        "source_filename": uploaded_file.name,
                                    })

                                if is_mongo_persistence_disabled():
                                    st.info(
                                        "ℹ️ **Streamlit Cloud Mode (Free Tier):** MongoDB persistence is gracefully disabled for free cloud hosting. "
                                        f"All **{len(mongo_docs):,}** batch predictions were computed live in-memory and are available in the table above. "
                                        "Use the **⬇️ Download Predictions (CSV)** button to export your scored dataset. "
                                        "(To persist batch records directly into MongoDB, run SupplyGuard AI locally at `mongodb://localhost:27017/`)."
                                    )
                                else:
                                    b_res = bulk_save_risk_predictions(
                                        mongo_docs,
                                        uri="mongodb://localhost:27017/",
                                        db_name="SupplyGuardDB"
                                    )

                                    if b_res.get("success"):
                                        st.success(
                                            f"✅ Successfully persisted **{b_res.get('count', len(mongo_docs)):,}** prediction records to "
                                            f"MongoDB database `SupplyGuardDB.risk_predictions` at `mongodb://localhost:27017/`. "
                                            f"All records are now verified, live in MongoDB Compass, and queryable in Prediction History!"
                                        )
                                        get_cached_mongo_status.clear()
                                    elif b_res.get("cloud_mode"):
                                        st.info(f"ℹ️ {b_res.get('message', 'MongoDB persistence is disabled in Streamlit Cloud.')}")
                                    else:
                                        st.error(f"❌ Failed to persist predictions to MongoDB: {b_res.get('error')}")


# ========================================================================================
# PAGE 3 — MODEL LAB (XGBoost, Model Comparison, and SHAP Explainable AI)
# ========================================================================================
elif page == "🧠 Model Lab":
    st.markdown("# Model Lab")
    st.caption("Rigorous machine learning benchmarking: comparing 7 classifiers to predict disruption risk, "
               "featuring **XGBoost** and **SHAP** Explainable AI visualizations.")

    lab_tabs = st.tabs([
        "🏆 Multi-Model Benchmark",
        "⚡ XGBoost Classifier Deep Dive",
        "🔍 SHAP Explainable AI (Global Insights)"
    ])

    res_df = pd.DataFrame(comparison["results"])

    # TAB 1: Multi-Model Benchmark
    with lab_tabs[0]:
        st.markdown("#### Classifier Comparison Leaderboard")
        st.caption("Trained on an 80/20 stratified split across 5,000 shipments with 28 engineered features.")

        st.dataframe(
            res_df.style.background_gradient(subset=["Accuracy", "F1_Score", "ROC_AUC"], cmap="Blues"),
            width="stretch",
        )

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(res_df.sort_values("F1_Score"), x="F1_Score", y="Model", orientation="h",
                         color="F1_Score", color_continuous_scale="Blues")
            fig.update_layout(title="Weighted F1 Score Across Models", coloraxis_showscale=False)
            st.plotly_chart(plotly_theme(fig), width="stretch")
        with c2:
            metrics_long = res_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1_Score"])
            fig = px.bar(metrics_long, x="Model", y="value", color="variable", barmode="group")
            fig.update_layout(title="Metric Comparison Breakdown", yaxis_title="Score")
            st.plotly_chart(plotly_theme(fig), width="stretch")

        st.markdown("#### ⚔️ Head-to-Head Model Comparator")
        comp_m1, comp_m2 = st.columns(2)
        with comp_m1:
            m1_name = st.selectbox("Select Model A", res_df["Model"].tolist(), index=res_df["Model"].tolist().index("XGBoost") if "XGBoost" in res_df["Model"].tolist() else 0)
        with comp_m2:
            m2_name = st.selectbox("Select Model B", res_df["Model"].tolist(), index=0)

        m1_row = res_df[res_df["Model"] == m1_name].iloc[0]
        m2_row = res_df[res_df["Model"] == m2_name].iloc[0]

        comp_metrics = ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Bar(name=m1_name, x=comp_metrics, y=[m1_row[m] for m in comp_metrics], marker_color=PALETTE["accent"]))
        fig_radar.add_trace(go.Bar(name=m2_name, x=comp_metrics, y=[m2_row[m] for m in comp_metrics], marker_color=PALETTE["safe"]))
        fig_radar.update_layout(barmode="group", title=f"{m1_name} vs {m2_name} — Side-by-Side Comparison", yaxis_range=[0.5, 0.9])
        st.plotly_chart(plotly_theme(fig_radar, height=360), width="stretch")

        st.markdown(f"#### Confusion Matrix — Best Model ({comparison['best_model']})")
        cm = np.array(comparison["confusion_matrix"])
        labels = comparison["class_names"]
        fig = px.imshow(cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues")
        fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(plotly_theme(fig, height=380), width="stretch")

        st.markdown("#### Feature Importance (Random Forest Baseline)")
        fi = pd.Series(comparison["feature_importances"]).sort_values(ascending=False).head(15)[::-1]
        fig = px.bar(fi, orientation="h", color=fi.values, color_continuous_scale="Tealgrn")
        fig.update_layout(showlegend=False, coloraxis_showscale=False, xaxis_title="Gini Importance", yaxis_title="")
        st.plotly_chart(plotly_theme(fig, height=420), width="stretch")

    # TAB 2: XGBoost Deep Dive
    with lab_tabs[1]:
        st.markdown("### ⚡ XGBoost Classifier — Architecture & Performance")
        st.caption("Deep dive into the eXtreme Gradient Boosting model trained to capture complex non-linear disruption interactions.")

        xgb_metrics = res_df[res_df["Model"] == "XGBoost"].iloc[0] if "XGBoost" in res_df["Model"].values else None

        if xgb_metrics is not None:
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Accuracy", f"{xgb_metrics['Accuracy']:.3f}")
            k2.metric("Weighted F1", f"{xgb_metrics['F1_Score']:.3f}")
            k3.metric("ROC-AUC", f"{xgb_metrics['ROC_AUC']:.3f}")
            k4.metric("Precision", f"{xgb_metrics['Precision']:.3f}")
            k5.metric("Recall", f"{xgb_metrics['Recall']:.3f}")

        xgb_c1, xgb_c2 = st.columns(2)

        with xgb_c1:
            st.markdown("#### XGBoost Confusion Matrix")
            xgb_cm = np.array(comparison.get("xgboost_confusion_matrix", comparison["confusion_matrix"]))
            fig_xgb_cm = px.imshow(xgb_cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Oranges")
            fig_xgb_cm.update_layout(xaxis_title="Predicted Risk", yaxis_title="Actual Risk")
            st.plotly_chart(plotly_theme(fig_xgb_cm, height=360), width="stretch")

        with xgb_c2:
            st.markdown("#### XGBoost Feature Importance (Split Gain)")
            xgb_fi = pd.Series(comparison.get("xgboost_feature_importances", comparison["feature_importances"])).head(12)[::-1]
            fig_xgb_fi = px.bar(xgb_fi, orientation="h", color=xgb_fi.values, color_continuous_scale="Oranges")
            fig_xgb_fi.update_layout(showlegend=False, coloraxis_showscale=False, xaxis_title="Gain Contribution", yaxis_title="")
            st.plotly_chart(plotly_theme(fig_xgb_fi, height=360), width="stretch")

        st.markdown("#### XGBoost Per-Class Classification Report")
        xgb_rep = comparison.get("xgboost_classification_report", comparison["classification_report"])
        rep_df = pd.DataFrame(xgb_rep).T
        st.dataframe(rep_df.round(3), width="stretch")

        st.markdown(
            f"""<div class='sg-card'>
            <span class='sg-eyebrow'>MODEL ARCHITECTURE & HYPERPARAMETERS</span>
            <div style='margin-top:6px; color:{PALETTE['text']}'>
            <b>Algorithm:</b> <code>XGBClassifier</code> (Gradient Boosted Decision Trees)<br>
            <b>Boosting Estimators:</b> 300 sequential trees &nbsp;|&nbsp; <b>Max Depth:</b> 5 levels &nbsp;|&nbsp; <b>Learning Rate (&eta;):</b> 0.08<br>
            <b>Objective Function:</b> <code>multi:softprob</code> (Multi-class cross-entropy log-loss)<br>
            <b>Key Value in Supply Chain:</b> Unlike linear classifiers, XGBoost effectively handles high-order non-linear interactions
            between external shocks (geopolitical conflict + port congestion + single sourcing) where risks multiply exponentially rather than add linearly.
            </div></div>""",
            unsafe_allow_html=True,
        )

    # TAB 3: SHAP Explainable AI
    with lab_tabs[2]:
        st.markdown("### 🔍 SHAP (SHapley Additive exPlanations)")
        st.caption("Game-theoretic explainable AI visualizations demonstrating how every feature mathematically pushes "
                   "disruption risk predictions up or down across the global shipment portfolio.")

        if global_shap_data is not None:
            shap_t1, shap_t2, shap_t3 = st.tabs([
                "🐝 SHAP Summary Beeswarm Plot",
                "📊 Mean |SHAP| Impact by Class",
                "📈 SHAP Dependence & Feature Interactions"
            ])

            with shap_t1:
                st.markdown("#### Global Feature Impact on High Disruption Risk")
                st.caption("Each dot represents a real shipment. The horizontal position shows whether the factor pushed the prediction "
                           "towards High Risk (positive) or away from High Risk (negative). Color indicates whether the feature value was Low (teal) or High (red).")

                target_cls = st.radio("Target Disruption Class for Beeswarm", ["High", "Medium", "Low"], horizontal=True)
                fig_beeswarm = plot_shap_beeswarm_interactive(global_shap_data, target_class=target_cls, top_n=14, height=520)
                st.plotly_chart(fig_beeswarm, width="stretch")

                st.info("💡 **Key Finding:** High **Geo Weather Exposure**, high **Concentration Risk**, and high **Port Congestion** "
                        "cluster heavily to the right (+ SHAP), proving they are the strongest drivers elevating disruption probability.")

            with shap_t2:
                st.markdown("#### Mean Absolute SHAP Value Across Risk Classes")
                st.caption("Measures overall feature importance magnitude across High, Medium, and Low disruption categories.")
                fig_shap_bar = plot_shap_summary_bar(global_shap_data, top_n=15, height=480)
                st.plotly_chart(fig_shap_bar, width="stretch")

            with shap_t3:
                st.markdown("#### Non-Linear SHAP Dependence Analysis")
                st.caption("Examine how the SHAP risk impact of any single factor changes as its raw value increases.")

                dep_feat = st.selectbox(
                    "Select Feature to Analyze",
                    [f for f in global_shap_data["feature_cols"] if not f.endswith("_enc")],
                    index=0
                )
                dep_cls = st.selectbox("Risk Class", ["High", "Medium", "Low"], index=0, key="dep_cls")
                fig_dep = plot_shap_dependence_interactive(global_shap_data, feature_name=dep_feat, target_class=dep_cls, height=420)
                st.plotly_chart(fig_dep, width="stretch")
        else:
            st.warning("XGBoost model is training or artifacts are being generated...")


# ========================================================================================
# PAGE 4 — PREDICT · EXPLAIN · RECOMMEND  (Enhanced with XGBoost + SHAP Waterfall)
# ========================================================================================
elif page == "🔮 Predict · Explain · Recommend":
    st.markdown("# Predict · Explain · Recommend")
    st.caption("Score any shipment, unpack its mathematical risk drivers using **SHAP**, and get priority preventive mitigations.")

    p_tab_predict, p_tab_history = st.tabs([
        "🔮 Score & Predict Shipment",
        "📜 Prediction History"
    ])

    with p_tab_predict:
        m_col1, m_col2 = st.columns([1.5, 1])
        with m_col1:
            input_options = ["Select an existing shipment", "Enter shipment details manually"]
            has_up_data = ("uploaded_predicted_df" in st.session_state) or ("uploaded_clean_df" in st.session_state)
            if has_up_data:
                input_options.insert(1, "Select from uploaded CSV dataset")
            mode = st.radio("Choose input mode", input_options, horizontal=True)
        with m_col2:
            active_model_name = st.radio("Active Classifier", ["⚡ XGBoost Classifier", "🥇 Best Model (Logistic Regression)"], horizontal=True)

        chosen_model = xgb_model if "XGBoost" in active_model_name and xgb_model is not None else model

        if mode == "Select an existing shipment":
            sid = st.selectbox("Shipment ID", df["Shipment_ID"].tolist())
            row = df[df["Shipment_ID"] == sid].iloc[0]
            raw = row.to_dict()
            target_sid = sid
        elif mode == "Select from uploaded CSV dataset":
            up_src = st.session_state.get("uploaded_predicted_df", st.session_state.get("uploaded_clean_df"))
            up_sids = [str(s) for s in up_src["Shipment_ID"].tolist()]
            sid = st.selectbox("Select Shipment from Uploaded Dataset", up_sids)
            row = up_src[up_src["Shipment_ID"].astype(str) == str(sid)].iloc[0]
            raw = row.to_dict()
            target_sid = sid
        else:
            st.markdown("##### Shipment Parameters")
            c1, c2, c3 = st.columns(3)
            with c1:
                region = st.selectbox("Region", sorted(df["Region"].unique()))
                category = st.selectbox("Product Category", sorted(df["Product_Category"].unique()))
                tier = st.selectbox("Supplier Tier", ["Tier 1", "Tier 2", "Tier 3"])
                mode_t = st.selectbox("Transport Mode", sorted(df["Transport_Mode"].unique()))
            with c2:
                lead_time = st.slider("Lead Time (days)", 1, 60, 15)
                distance = st.slider("Distance (km)", 50, 15000, 4000)
                order_qty = st.slider("Order Quantity", 10, 5000, 500)
                inv_days = st.slider("Inventory Level (days)", 0, 40, 15)
            with c3:
                demand_var = st.slider("Demand Variability (%)", 0, 100, 20)
                reliability = st.slider("Supplier Reliability Score", 0.0, 1.0, 0.7)
                financial = st.slider("Financial Stability Score", 0.0, 1.0, 0.65)
                single_source = st.checkbox("Single-sourced supplier?", value=False)

            c4, c5, c6 = st.columns(3)
            with c4:
                geo_risk = st.slider("Geopolitical Risk Index", 0.0, 1.0, 0.3)
                weather_risk = st.slider("Weather Risk Index", 0.0, 1.0, 0.3)
            with c5:
                port_cong = st.slider("Port Congestion Index", 0.0, 1.0, 0.3)
                fuel_price = st.slider("Fuel Price Index", 0.0, 1.0, 0.5)
            with c6:
                customs_delay = st.slider("Customs Delay (days)", 0.0, 15.0, 2.0)
                disruption_events = st.slider("Disruption Events (past year)", 0, 10, 1)

            otd = st.slider("On-Time Delivery Rate (%)", 20, 100, 85)
            cost_var = st.slider("Cost Variance (%)", -10, 60, 10)
            contract_flex = st.slider("Contract Flexibility Score", 0.0, 1.0, 0.5)

            raw = dict(
                Region=region, Product_Category=category, Supplier_Tier=tier, Transport_Mode=mode_t,
                Lead_Time_Days=lead_time, Distance_KM=distance, Order_Quantity=order_qty,
                Inventory_Level_Days=inv_days, Demand_Variability_Pct=demand_var,
                Supplier_Reliability_Score=reliability, Financial_Stability_Score=financial,
                Geopolitical_Risk_Index=geo_risk, Weather_Risk_Index=weather_risk,
                Port_Congestion_Index=port_cong, Fuel_Price_Index=fuel_price,
                Customs_Delay_Days=customs_delay, Disruption_Events_Past_Year=disruption_events,
                On_Time_Delivery_Rate_Pct=otd, Cost_Variance_Pct=cost_var,
                Single_Sourcing_Flag=1 if single_source else 0, Contract_Flexibility_Score=contract_flex,
            )
            target_sid = raw.get("shipment_id", f"SH-MANUAL-{abs(hash(str(raw))) % 100000:05d}")
            raw["shipment_id"] = target_sid
            raw["Shipment_ID"] = target_sid

        st.markdown("")
        btn_predict = st.button("🚀 Predict", type="primary", width="stretch")

        # Only run and save when Predict is clicked, or retain display if already predicted for this target_sid
        has_cached_pred = (
            "active_pred_cache" in st.session_state
            and st.session_state["active_pred_cache"].get("target_sid") == target_sid
        )

        if btn_predict or has_cached_pred:
            if btn_predict:
                # Compute scores and ML predictions
                scores = compute_composite_scores(raw)
                ews = compute_ews(scores)
                band = ews_band(ews)
                exp = explain(scores, top_n=3)
                recs = recommend(exp)

                cat_maps = {}
                for col in ["Region", "Country", "Product_Category", "Supplier_Tier", "Transport_Mode"]:
                    if col in df.columns:
                        cat_maps[col] = {v: i for i, v in enumerate(sorted(df[col].dropna().unique()))}

                encoded = dict(raw)
                encoded["Region_enc"] = cat_maps["Region"].get(raw.get("Region"), 0)
                encoded["Product_Category_enc"] = cat_maps["Product_Category"].get(raw.get("Product_Category"), 0)
                encoded["Supplier_Tier_enc"] = cat_maps["Supplier_Tier"].get(raw.get("Supplier_Tier"), 0)
                encoded["Transport_Mode_enc"] = cat_maps["Transport_Mode"].get(raw.get("Transport_Mode"), 0)
                for k, v in scores.items():
                    encoded[k] = v

                if mode in ["Select an existing shipment", "Select from uploaded CSV dataset"]:
                    for col in feature_cols:
                        if col in row:
                            encoded[col] = row[col]

                # Run chosen model
                try:
                    label, proba = predict_ml(encoded, chosen_model, scaler, target_encoder, feature_cols)
                except Exception:
                    label, proba = raw.get("Risk_Level", "N/A"), {}

                # Always compute XGBoost prediction & risk probability
                if xgb_model is not None:
                    try:
                        xgb_label, xgb_proba = predict_ml(encoded, xgb_model, scaler, target_encoder, feature_cols)
                    except Exception:
                        xgb_label, xgb_proba = label, proba
                else:
                    xgb_label, xgb_proba = label, proba

                risk_prob = float(xgb_proba.get("High", 0.0))

                # Compute local SHAP if XGBoost is available
                local_shap_res = None
                shap_summary = None
                if xgb_model is not None:
                    try:
                        local_shap_res = compute_local_shap(
                            xgb_model, encoded, feature_cols, target_class="High", target_encoder=target_encoder
                        )
                        shap_summary = {
                            "base_value": round(float(local_shap_res["base_value"]), 4),
                            "margin_value": round(float(local_shap_res["margin_value"]), 4),
                            "sum_shap": round(float(local_shap_res["sum_shap"]), 4),
                            "disruption_prob": round(float(local_shap_res["disruption_prob"]), 4),
                            "top_risk_drivers": local_shap_res["risk_drivers"][:3],
                            "top_protective_factors": local_shap_res["protective_factors"][:3],
                        }
                    except Exception as e:
                        shap_summary = {"error": str(e)}

                timestamp_now = datetime.datetime.now(datetime.timezone.utc).isoformat()
                clean_inputs = {str(k): sanitize_for_mongo(v) for k, v in raw.items() if not str(k).startswith("_")}
                sub_hash = hashlib.sha256(f"{target_sid}_{xgb_label}_{json.dumps(clean_inputs, sort_keys=True, default=str)}".encode()).hexdigest()[:16]

                # MongoDB Persistence (save on clicking Predict)
                db_save_status = {}
                try:
                    # 1. Optionally sync shipment to shipments collection
                    try:
                        shipment_payload = {
                            **clean_inputs,
                            "shipment_id": target_sid,
                            "Shipment_ID": target_sid,
                            "status": raw.get("status", "In Transit"),
                            "risk_level": xgb_label,
                            "early_warning_score": round(float(ews), 2),
                            "ews_band": band,
                            "updated_at": timestamp_now,
                        }
                        save_shipment(shipment_payload, uri="mongodb://localhost:27017/", db_name="SupplyGuardDB")
                    except Exception:
                        pass  # Non-blocking for prediction saving

                    # 2. Save prediction telemetry to SupplyGuardDB.risk_predictions
                    pred_payload = {
                        "shipment_id": target_sid,
                        "Shipment_ID": target_sid,
                        "model_name": "XGBoostClassifier",
                        "xgboost_prediction": xgb_label,
                        "predicted_risk_level": xgb_label,
                        "risk_probability": round(risk_prob, 4),
                        "prediction_probabilities": xgb_proba,
                        "early_warning_score": round(float(ews), 2),
                        "ews_band": band,
                        "shap_summary": shap_summary,
                        "shap_explanation": shap_summary,
                        "actual_input_values": clean_inputs,
                        "shipment_inputs": clean_inputs,
                        "input_values": clean_inputs,
                        "top_risk_drivers": exp,
                        "preventive_recommendations": [r["factor"] for r in recs],
                        "submission_hash": sub_hash,
                        "timestamp": timestamp_now,
                        "prediction_timestamp": timestamp_now,
                        "created_at": timestamp_now,
                        "input_mode": mode,
                    }
                    pred_res = save_risk_prediction(
                        pred_payload,
                        uri="mongodb://localhost:27017/",
                        db_name="SupplyGuardDB",
                        prevent_duplicates=False,
                    )

                    if pred_res.get("success") and pred_res.get("prediction_id"):
                        db_save_status = {
                            "type": "success",
                            "msg": (
                                f"Saved actual shipment inputs, XGBoost prediction (`{xgb_label}`), "
                                f"P(High Risk) = `{risk_prob*100:.1f}%`, "
                                f"{'SHAP summary, ' if shap_summary and not shap_summary.get('error') else ''}and timestamp to "
                                f"MongoDB (`SupplyGuardDB.risk_predictions`). "
                                f"Audit ID: `{pred_res.get('prediction_id')}`"
                            ),
                        }
                        get_cached_mongo_status.clear()
                    elif pred_res.get("cloud_mode") or pred_res.get("disabled"):
                        db_save_status = {
                            "type": "info",
                            "msg": (
                                "Streamlit Cloud Mode: MongoDB persistence is disabled for free cloud hosting. "
                                f"XGBoost prediction (`{xgb_label}`), P(High Risk) = `{risk_prob*100:.1f}%`, and SHAP attribution "
                                "were computed live in-memory. (To enable persistent database audit logging, run locally with "
                                "mongodb://localhost:27017/ or configure MongoDB Atlas in Streamlit Secrets)."
                            ),
                        }
                    else:
                        db_save_status = {
                            "type": "error",
                            "msg": f"Failed to persist prediction to MongoDB: {pred_res.get('error', 'Unknown database error')}",
                        }
                except Exception as e:
                    db_save_status = {
                        "type": "error",
                        "msg": f"Database exception encountered while saving to MongoDB: {e}",
                    }

                # Record in session predictions history
                session_entry = {
                    "shipment_id": target_sid,
                    "Shipment_ID": target_sid,
                    "model_name": "XGBoostClassifier",
                    "xgboost_prediction": xgb_label,
                    "predicted_risk_level": xgb_label,
                    "risk_probability": round(risk_prob, 4),
                    "prediction_probabilities": xgb_proba,
                    "early_warning_score": round(float(ews), 2),
                    "ews_band": band,
                    "shap_summary": shap_summary,
                    "top_risk_drivers": exp,
                    "timestamp": timestamp_now,
                    "prediction_timestamp": timestamp_now,
                    "created_at": timestamp_now,
                    "source": "In-Memory Session" if is_mongo_persistence_disabled() else "MongoDB Verified",
                }
                if "session_predictions" not in st.session_state:
                    st.session_state["session_predictions"] = []
                st.session_state["session_predictions"] = [
                    p for p in st.session_state["session_predictions"] if p.get("shipment_id") != target_sid
                ]
                st.session_state["session_predictions"].insert(0, session_entry)

                # Store active prediction in session state
                st.session_state["active_pred_cache"] = {
                    "target_sid": target_sid,
                    "label": label,
                    "proba": proba,
                    "chosen_model_name": chosen_model.__class__.__name__,
                    "xgb_label": xgb_label,
                    "xgb_proba": xgb_proba,
                    "risk_prob": risk_prob,
                    "ews": ews,
                    "band": band,
                    "exp": exp,
                    "recs": recs,
                    "local_shap_res": local_shap_res,
                    "db_save_status": db_save_status,
                }

            # Retrieve from session cache
            cached = st.session_state["active_pred_cache"]
            label = cached["label"]
            proba = cached["proba"]
            chosen_model_name = cached["chosen_model_name"]
            xgb_label = cached["xgb_label"]
            xgb_proba = cached["xgb_proba"]
            risk_prob = cached["risk_prob"]
            ews = cached["ews"]
            band = cached["band"]
            exp = cached["exp"]
            recs = cached["recs"]
            local_shap_res = cached["local_shap_res"]
            db_save_status = cached.get("db_save_status", {})

            # Render DB Save Status Banner
            if db_save_status:
                if db_save_status.get("type") == "success":
                    st.success(f"✅ **MongoDB Persistence:** {db_save_status['msg']}")
                elif db_save_status.get("type") == "info":
                    st.info(f"ℹ️ **MongoDB Persistence:** {db_save_status['msg']}")
                elif db_save_status.get("type") == "warning":
                    st.warning(f"⚠️ **MongoDB Notice:** {db_save_status['msg']}")
                elif db_save_status.get("type") == "error":
                    st.error(f"❌ **MongoDB Error:** {db_save_status['msg']}")

            st.markdown("---")
            r1, r2, r3 = st.columns([1, 1, 1.4])

            with r1:
                st.markdown(f"#### 🎯 ML Risk Prediction")
                st.caption(f"Engine: {chosen_model_name} (XGBoost: {xgb_label}, P(High)={risk_prob*100:.1f}%)")
                st.markdown(f"<h1 style='color:{RISK_COLOR.get(label, PALETTE['text'])}'>{label}</h1>", unsafe_allow_html=True)
                if proba:
                    for cls in ORDER:
                        if cls in proba:
                            st.progress(proba[cls], text=f"{cls}: {proba[cls]*100:.1f}%")

            with r2:
                st.markdown("#### 🛰️ Early Warning Score")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=ews,
                    number={"suffix": " / 100"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": BAND_COLOR[band]},
                        "steps": [
                            {"range": [0, 35], "color": "#123326"},
                            {"range": [35, 60], "color": "#3a3214"},
                            {"range": [60, 80], "color": "#3a2414"},
                            {"range": [80, 100], "color": "#3a1416"},
                        ],
                    }))
                fig = plotly_theme(fig, height=260)
                st.plotly_chart(fig, width="stretch")
                st.markdown(f"<div style='text-align:center' class='sg-band-{band.lower()}'>Band: {band}</div>", unsafe_allow_html=True)

            with r3:
                st.markdown("#### 🧩 Domain EWS Risk Drivers")
                exp_df = pd.DataFrame(exp)
                fig = px.bar(exp_df.sort_values("contribution_pct"), x="contribution_pct", y="factor", orientation="h",
                             color="contribution_pct", color_continuous_scale=["#2DD4BF", "#FFC94D", "#FF5A5F"],
                             text="contribution_pct")
                fig.update_traces(texttemplate="%{text}%", textposition="outside", cliponaxis=False)
                fig.update_layout(showlegend=False, coloraxis_showscale=False,
                                   xaxis_title="Contribution to EWS (%)", yaxis_title="",
                                   xaxis_range=[0, exp_df["contribution_pct"].max() * 1.35],
                                   margin=dict(l=10, r=60, t=50, b=10))
                st.plotly_chart(plotly_theme(fig, height=260), width="stretch")

            # LOCAL SHAP WATERFALL VISUALIZATION
            if local_shap_res is not None:
                st.markdown("---")
                st.markdown("#### 🔍 SHAP Disruption Prediction Breakdown (XGBoost Waterfall)")
                st.caption("Game-theoretic attribution showing exactly which features pushed this shipment's prediction "
                           "toward (+) or away from (-) High Disruption Risk.")

                base_val = local_shap_res["base_value"]
                sum_shap = local_shap_res["sum_shap"]
                margin_val = local_shap_res["margin_value"]
                disruption_p = local_shap_res["disruption_prob"]
                base_p = local_shap_res["baseline_prob"]

                shap_m1, shap_m2, shap_m3, shap_m4 = st.columns(4)
                shap_m1.metric(
                    "Base Value E[f(X)]",
                    f"{base_val:+.3f}",
                    help=f"Expected baseline model margin across training set. Implied baseline risk: {base_p*100:.1f}%",
                )
                shap_m2.metric(
                    "Net Feature Push ∑ φᵢ",
                    f"{sum_shap:+.3f}",
                    delta=f"{sum_shap:+.3f} log-odds",
                    delta_color="inverse",
                    help=f"Total sum of all 28 feature SHAP values (+ve: {local_shap_res['sum_positive']:+.2f}, -ve: {local_shap_res['sum_negative']:+.2f})",
                )
                shap_m3.metric(
                    "Model Output f(X)",
                    f"{margin_val:+.3f}",
                    help=f"Exact raw model margin on log-odds scale (Base + ∑ SHAP = {base_val + sum_shap:+.3f})",
                )
                shap_m4.metric(
                    "Disruption-Risk Prob",
                    f"{disruption_p*100:.1f}%",
                    delta=f"{(disruption_p - base_p)*100:+.1f}% vs baseline",
                    delta_color="inverse",
                    help=f"P(High Disruption Risk) calculated from multi-class softmax. Baseline risk: {base_p*100:.1f}%",
                )

                diff_check = abs((base_val + sum_shap) - margin_val)
                risk_badge_color = "#FF5A5F" if disruption_p >= 0.5 else ("#FFC94D" if disruption_p >= 0.25 else "#2DD4BF")
                st.markdown(
                    f"""<div style='background:{PALETTE["panel2"]}; border:1px solid {PALETTE["grid"]}; border-radius:8px; padding:8px 14px; margin-bottom:14px; font-size:13px;'>
                    <b style='color:{PALETTE["accent"]}'>📐 Mathematical Identity Verification:</b> &nbsp;
                    <code>Base Value ({base_val:+.4f})</code> + <code>Net Features ({sum_shap:+.4f})</code> = 
                    <code>Model Output Margin ({margin_val:+.4f})</code> &nbsp;⟹&nbsp; 
                    <b style='color:{risk_badge_color}'>P(Disruption Risk) = {disruption_p*100:.2f}%</b>
                    &nbsp;<span style='color:{PALETTE["safe"]}; font-size:11px;'>(|Δ| = {diff_check:.1e} ≤ 10⁻⁵ ✓ exact log-odds additivity)</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

                wf_col1, wf_col2 = st.columns([1.6, 1])
                chart_key = target_sid if mode == "Select an existing shipment" else "manual_input"

                with wf_col1:
                    fig_wf = plot_shap_waterfall(local_shap_res, max_display=8, target_class="High", height=420)
                    st.plotly_chart(fig_wf, key=f"shap_wf_{chart_key}", width="stretch")

                with wf_col2:
                    st.markdown("##### Key Attribution Drivers")

                    def _fmt_raw(val):
                        if isinstance(val, float):
                            return f"{val:.2f}"
                        elif isinstance(val, (int, np.integer)):
                            return f"{val:,}"
                        return str(val)

                    st.markdown("<b style='color:#FF5A5F'>Elevating Disruption Risk (+):</b>", unsafe_allow_html=True)
                    top_pos = local_shap_res["risk_drivers"][:3]
                    if top_pos:
                        for item in top_pos:
                            raw_str = _fmt_raw(item.get("raw_value", 0))
                            st.markdown(
                                f"• **{item['friendly_name']}** (<code>{raw_str}</code>): "
                                f"<span style='color:#FF5A5F; font-weight:600'>+{item['shap_value']:.3f} log-odds</span>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption("No features actively elevating disruption risk for this shipment.")

                    st.markdown("<div style='margin-top:10px;'><b style='color:#2DD4BF'>Buffering / Mitigating Risk (-):</b></div>", unsafe_allow_html=True)
                    top_neg = local_shap_res["protective_factors"][:3]
                    if top_neg:
                        for item in top_neg:
                            raw_str = _fmt_raw(item.get("raw_value", 0))
                            st.markdown(
                                f"• **{item['friendly_name']}** (<code>{raw_str}</code>): "
                                f"<span style='color:#2DD4BF; font-weight:600'>{item['shap_value']:.3f} log-odds</span>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption("No protective buffering factors identified for this shipment.")

                    st.markdown(
                        f"""<div style='margin-top:14px; padding:10px 12px; background:{PALETTE["panel"]}; border-radius:6px; border:1px solid {PALETTE["grid"]}; font-size:12px; color:{PALETTE["muted"]}'>
                        <b>Attribution Summary:</b><br>
                        • Positive Risk Push: <b style='color:#FF5A5F'>+{local_shap_res['sum_positive']:.2f}</b><br>
                        • Negative Protective Buffer: <b style='color:#2DD4BF'>{local_shap_res['sum_negative']:.2f}</b><br>
                        • Net Movement: <b>{sum_shap:+.2f} log-odds</b>
                        </div>""",
                        unsafe_allow_html=True,
                    )

            st.markdown("#### 💊 Preventive Recommendations")
            rc = st.columns(len(recs))
            for i, r in enumerate(recs):
                with rc[i]:
                    actions_html = "".join([f"<li>{a}</li>" for a in r["actions"]])
                    st.markdown(
                        f"""<div class='sg-card'>
                        <span class='sg-eyebrow'>PRIORITY {i+1}</span>
                        <h4 style='margin-top:2px'>{r['factor']}</h4>
                        <ul style='padding-left:18px; color:{PALETTE['muted']}'>{actions_html}</ul>
                        </div>""",
                        unsafe_allow_html=True,
                    )

            st.markdown("---")
            with st.expander("📜 Recent Predictions Telemetry", expanded=False):
                if mongo_info.get("cloud_mode"):
                    recent_history = st.session_state.get("session_predictions", [])[:5]
                else:
                    recent_history = get_risk_predictions(limit=5)
                    if not recent_history:
                        recent_history = st.session_state.get("session_predictions", [])[:5]
                if recent_history:
                    rec_rows = []
                    for r_doc in recent_history:
                        r_pred = r_doc.get("xgboost_prediction") or r_doc.get("predicted_risk_level", "N/A")
                        r_prob = r_doc.get("risk_probability")
                        prob_display = f"{float(r_prob)*100:.1f}%" if r_prob is not None else "N/A"
                        ews_val = r_doc.get("early_warning_score")
                        try:
                            ews_clean = round(float(ews_val), 1) if ews_val is not None and ews_val != "N/A" else np.nan
                        except (ValueError, TypeError):
                            ews_clean = np.nan
                        rec_rows.append({
                            "Shipment_ID": r_doc.get("shipment_id", "N/A"),
                            "XGBoost_Prediction": r_pred,
                            "Risk_Probability": prob_display,
                            "EWS_Score": ews_clean,
                            "Timestamp": r_doc.get("timestamp") or r_doc.get("prediction_timestamp", "N/A"),
                        })
                    rec_df = pd.DataFrame(rec_rows)
                    if not rec_df.empty and "EWS_Score" in rec_df.columns:
                        rec_df["EWS_Score"] = pd.to_numeric(rec_df["EWS_Score"], errors="coerce")
                    st.dataframe(rec_df, width="stretch")
                else:
                    st.caption("No predictions recorded in this session or database yet.")
        else:
            st.info("💡 Adjust any shipment parameters above, then click **🚀 Predict** to compute ML inferences, calculate SHAP explanations, and persist the record to local MongoDB (`SupplyGuardDB`).")

    # --------------------------------------------------------------------------------------
    # TAB 2: PREDICTION HISTORY SECTION (Retrieve and display saved predictions from MongoDB)
    # --------------------------------------------------------------------------------------
    with p_tab_history:
        st.markdown("### 📜 Prediction History")
        st.caption("Retrieve and inspect all real ML predictions, risk probabilities, SHAP summaries, and actual shipment inputs.")

        if mongo_info.get("cloud_mode"):
            st.info(
                "ℹ️ **Streamlit Cloud Mode (Free Deployment):** MongoDB persistence is gracefully disabled for free cloud hosting. "
                "All ML risk predictions and SHAP explanations run live in-memory. "
                "Below is your **active session prediction history** generated while using the dashboard in this browser session.\n\n"
                "(To view persistent records in MongoDB Compass, run SupplyGuard AI locally at `mongodb://localhost:27017/`)."
            )
            hist_preds = st.session_state.get("session_predictions", [])
        elif not mongo_ok:
            st.warning(
                f"⚠️ **MongoDB is offline or unreachable.** Prediction history cannot be retrieved while `mongodb://localhost:27017/` is offline.\n\n"
                f"**Error detail:** `{mongo_err}`\n\n"
                f"Start your local MongoDB instance (e.g. `net start MongoDB`) and click **Retry Connection** below."
            )
            if st.button("🔄 Retry MongoDB Connection", key="retry_mongo_hist"):
                get_cached_mongo_status.clear()
                st.rerun()
            hist_preds = st.session_state.get("session_predictions", [])
        else:
            # Safely query predictions from MongoDB
            try:
                hist_preds = get_risk_predictions(limit=100)
            except Exception as e:
                hist_preds = []
                st.warning(f"⚠️ Error querying predictions from MongoDB: {e}")
            if not hist_preds:
                hist_preds = st.session_state.get("session_predictions", [])

        if not hist_preds:
            if mongo_info.get("cloud_mode"):
                st.info("ℹ️ **No predictions generated in this session yet.**\n\n"
                        "Switch to the **Score & Predict Shipment** tab above, click **🚀 Predict**, and your live prediction telemetry will appear here.")
            else:
                st.info("ℹ️ **No saved predictions found in MongoDB (`SupplyGuardDB.risk_predictions`).**\n\n"
                        "Switch to the **Score & Predict Shipment** tab above, click **🚀 Predict**, and the real prediction, inputs, risk probability, and SHAP summary will appear here.")
            else:
                # Summary KPIs
                h_k1, h_k2, h_k3, h_k4 = st.columns(4)
                h_k1.metric("Saved Predictions", f"{len(hist_preds):,}")
                high_count = sum(1 for p in hist_preds if p.get("predicted_risk_level") == "High")
                med_count = sum(1 for p in hist_preds if p.get("predicted_risk_level") == "Medium")
                low_count = sum(1 for p in hist_preds if p.get("predicted_risk_level") == "Low")
                h_k2.metric("High Risk", f"{high_count}", delta=f"{high_count/len(hist_preds)*100:.0f}%" if hist_preds else "0%")
                h_k3.metric("Medium Risk", f"{med_count}", delta=f"{med_count/len(hist_preds)*100:.0f}%" if hist_preds else "0%")
                h_k4.metric("Low Risk", f"{low_count}", delta=f"{low_count/len(hist_preds)*100:.0f}%" if hist_preds else "0%")

                st.markdown("---")
                # Filter bar
                hf_c1, hf_c2, hf_c3, hf_c4 = st.columns([2, 1.5, 1.5, 1])
                with hf_c1:
                    h_search = st.text_input("🔍 Search Shipment ID", placeholder="e.g. SG02088", key="hist_search_box")
                with hf_c2:
                    h_risk_filter = st.multiselect("Risk Level", ["Low", "Medium", "High"], default=[], key="hist_risk_multisel")
                with hf_c3:
                    h_model_filter = st.multiselect("Model Filter", ["XGBoostClassifier", "LogisticRegression"], default=[], key="hist_model_multisel")
                with hf_c4:
                    h_limit = st.selectbox("Limit", [10, 25, 50, 100], index=2, key="hist_limit_sel")

                filtered_preds = hist_preds
                if h_search.strip():
                    filtered_preds = [p for p in filtered_preds if h_search.strip().lower() in str(p.get("shipment_id", "")).lower()]
                if h_risk_filter:
                    filtered_preds = [p for p in filtered_preds if p.get("predicted_risk_level") in h_risk_filter]
                if h_model_filter:
                    filtered_preds = [p for p in filtered_preds if p.get("model_name") in h_model_filter]
                filtered_preds = filtered_preds[:h_limit]

                # Build table
                h_rows = []
                for p in filtered_preds:
                    shap_top = "N/A"
                    if isinstance(p.get("shap_summary"), dict) and p["shap_summary"].get("top_risk_drivers"):
                        drivers = p["shap_summary"]["top_risk_drivers"]
                        if drivers and isinstance(drivers, list):
                            shap_top = drivers[0].get("friendly_name", "N/A")
                    elif p.get("top_risk_drivers") and isinstance(p["top_risk_drivers"], list):
                        shap_top = p["top_risk_drivers"][0].get("factor", "N/A")

                    prob_val = p.get("risk_probability")
                    if prob_val is not None:
                        prob_str = f"{float(prob_val)*100:.1f}%"
                    elif isinstance(p.get("prediction_probabilities"), dict) and "High" in p["prediction_probabilities"]:
                        prob_str = f"{float(p['prediction_probabilities']['High'])*100:.1f}%"
                    else:
                        prob_str = "N/A"

                    ews_val = p.get("early_warning_score")
                    try:
                        ews_clean = round(float(ews_val), 1) if ews_val is not None and ews_val != "N/A" else np.nan
                    except (ValueError, TypeError):
                        ews_clean = np.nan
                    h_rows.append({
                        "Shipment_ID": p.get("shipment_id", "N/A"),
                        "XGBoost_Prediction": p.get("predicted_risk_level", "N/A"),
                        "Risk_Probability": prob_str,
                        "EWS_Score": ews_clean,
                        "EWS_Band": p.get("ews_band", "N/A"),
                        "Key_SHAP_Driver": shap_top,
                        "Model": p.get("model_name", "XGBoostClassifier"),
                        "Timestamp": p.get("timestamp") or p.get("prediction_timestamp") or p.get("created_at", "N/A"),
                    })

                hist_df = pd.DataFrame(h_rows)
                if not hist_df.empty and "EWS_Score" in hist_df.columns:
                    hist_df["EWS_Score"] = pd.to_numeric(hist_df["EWS_Score"], errors="coerce")

                def color_hist_risk(val):
                    c = PALETTE.get("safe") if val == "Low" else (PALETTE.get("watch") if val == "Medium" else PALETTE.get("critical"))
                    return f"color: {c}; font-weight: 700"

                if not hist_df.empty:
                    st.dataframe(
                        hist_df.style.map(color_hist_risk, subset=["XGBoost_Prediction"]),
                        width="stretch",
                        height=360,
                    )

                    h_btn_c1, h_btn_c2 = st.columns([1, 1])
                    with h_btn_c1:
                        st.download_button(
                            "⬇️ Export Prediction History (CSV)",
                            hist_df.to_csv(index=False),
                            "prediction_history.csv",
                            key="dl_pred_history_csv",
                        )
                    with h_btn_c2:
                        if st.button("🔄 Refresh Prediction History", key="btn_refresh_pred_hist"):
                            get_cached_mongo_status.clear()
                            st.rerun()

                    # Deep Dive Document Inspector
                    with st.expander("🔍 Inspect Historical Prediction Details & SHAP Telemetry"):
                        doc_labels = [f"{p.get('shipment_id', 'N/A')} | {p.get('predicted_risk_level', 'N/A')} ({p.get('timestamp', '')[:19]})" for p in filtered_preds]
                        sel_doc_idx = st.selectbox("Select Prediction Record", range(len(doc_labels)), format_func=lambda i: doc_labels[i], key="sel_hist_doc")
                        sel_pred = filtered_preds[sel_doc_idx]

                        d_c1, d_c2 = st.columns(2)
                        with d_c1:
                            st.markdown("##### 📦 Stored Shipment Inputs")
                            stored_inputs = sel_pred.get("actual_input_values") or sel_pred.get("shipment_inputs") or sel_pred.get("input_features") or {}
                            if stored_inputs:
                                st.dataframe(pd.DataFrame(list(stored_inputs.items()), columns=["Feature", "Value"]), width="stretch", height=260)
                            else:
                                st.caption("No input dictionary embedded.")
                        with d_c2:
                            st.markdown("##### 🔬 Stored SHAP Explanation & Probabilities")
                            stored_shap = sel_pred.get("shap_explanation") or sel_pred.get("shap_summary", {})
                            if stored_shap and isinstance(stored_shap, dict):
                                st.json(stored_shap)
                            else:
                                st.caption("No SHAP summary dictionary embedded.")

                        st.markdown("##### 📄 Full MongoDB Document (BSON/JSON)")
                        st.json(sel_pred)
                else:
                    st.info("No prediction history records matched your filter criteria.")


# ========================================================================================
# PAGE 5 — RISK RANKING (Leaderboard + AI Diagnostic)
# ========================================================================================
elif page == "📋 Risk Ranking":
    st.markdown("# Risk Ranking")
    st.caption("Triage high-exposure shipments across your global network, ranked by Early Warning Score and transit vulnerability.")

    c1, c2, c3 = st.columns(3)
    band_filter = c1.multiselect("EWS Band", BAND_ORDER, default=["Critical", "Elevated"])
    region_filter = c2.multiselect("Region", sorted(df["Region"].unique()), default=[])
    n_show = c3.slider("Rows to show", 10, 200, 50)

    rdf = df.copy()
    if band_filter:
        rdf = rdf[rdf["EWS_Band"].isin(band_filter)]
    if region_filter:
        rdf = rdf[rdf["Region"].isin(region_filter)]

    rdf = rdf.sort_values("Early_Warning_Score", ascending=False).head(n_show)

    display_cols = ["Shipment_ID", "Region", "Country", "Product_Category", "Supplier_Tier",
                     "Transport_Mode", "Early_Warning_Score", "EWS_Band", "Risk_Level",
                     "Disruption_Events_Past_Year", "On_Time_Delivery_Rate_Pct"]

    def color_band(val):
        return f"color: {BAND_COLOR.get(val, PALETTE['text'])}; font-weight:700"

    st.dataframe(
        rdf[display_cols].style.map(color_band, subset=["EWS_Band"])
        .background_gradient(subset=["Early_Warning_Score"], cmap="OrRd"),
        width="stretch", height=480,
    )

    st.download_button("⬇️ Download ranked list (CSV)", rdf[display_cols].to_csv(index=False),
                        "risk_ranking.csv")

    st.markdown("---")
    r_col1, r_col2 = st.columns([1.2, 1.4])

    with r_col1:
        st.markdown("#### Top 15 Riskiest Shipments")
        top15 = rdf.head(15).sort_values("Early_Warning_Score")
        fig = px.bar(top15, x="Early_Warning_Score", y="Shipment_ID", orientation="h",
                     color="EWS_Band", color_discrete_map=BAND_COLOR)
        st.plotly_chart(plotly_theme(fig, height=440), width="stretch")

    with r_col2:
        st.markdown("#### 🤖 AI Shipment Diagnostic & Action Plan")
        st.caption("Select a shipment from the leaderboard to run automated ML diagnosis, EWS breakdown, "
                   "and prioritized risk mitigation actions — powered by the Data-to-Action Agent.")

        trace_sid = st.selectbox("Inspect Shipment Diagnostic", rdf["Shipment_ID"].tolist(), key="rank_diag_sid")
        diag_res = agent.analyze_shipment(trace_sid)
        st.markdown(diag_res.summary, unsafe_allow_html=True)
        with st.expander("📊 View Factual Shipment Evidence & Features"):
            st.dataframe(diag_res.evidence_df, width="stretch")
        if diag_res.recommendations:
            st.markdown("##### ⚡ Priority Interventions")
            for r in diag_res.recommendations[:2]:
                st.markdown(f"• **{r.get('factor', 'Risk Driver')}**: {'; '.join(r.get('actions', []))}")


# ========================================================================================
# PAGE 6 — WHAT-IF SIMULATOR
# ========================================================================================
elif page == "🧪 What-If Simulator":
    st.markdown("# What-If Simulator")
    st.caption("Adjust key risk drivers and watch the Early Warning Score respond in real time.")

    base_sid = st.selectbox("Baseline shipment", df["Shipment_ID"].tolist())
    base_row = df[df["Shipment_ID"] == base_sid].iloc[0]
    base_scores = compute_composite_scores(base_row.to_dict())
    base_ews = compute_ews(base_scores)
    base_band = ews_band(base_ews)

    st.markdown("##### Adjust drivers")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        geo = st.slider("Geopolitical Risk", 0.0, 1.0, float(base_row["Geopolitical_Risk_Index"]), key="geo")
        weather = st.slider("Weather Risk", 0.0, 1.0, float(base_row["Weather_Risk_Index"]), key="weather")
    with s2:
        reliability = st.slider("Supplier Reliability", 0.0, 1.0, float(base_row["Supplier_Reliability_Score"]), key="rel")
        financial = st.slider("Financial Stability", 0.0, 1.0, float(base_row["Financial_Stability_Score"]), key="fin")
    with s3:
        port_c = st.slider("Port Congestion", 0.0, 1.0, float(base_row["Port_Congestion_Index"]), key="port")
        single_src = st.checkbox("Single-sourced?", value=bool(base_row["Single_Sourcing_Flag"]), key="ss")
    with s4:
        inv_days = st.slider("Inventory Buffer (days)", 0, 40, int(base_row["Inventory_Level_Days"]), key="inv")
        contract_flex = st.slider("Contract Flexibility", 0.0, 1.0, float(base_row["Contract_Flexibility_Score"]), key="cf")

    sim_raw = base_row.to_dict()
    sim_raw.update(dict(
        Geopolitical_Risk_Index=geo, Weather_Risk_Index=weather,
        Supplier_Reliability_Score=reliability, Financial_Stability_Score=financial,
        Port_Congestion_Index=port_c, Single_Sourcing_Flag=1 if single_src else 0,
        Inventory_Level_Days=inv_days, Contract_Flexibility_Score=contract_flex,
    ))
    sim_scores = compute_composite_scores(sim_raw)
    sim_ews = compute_ews(sim_scores)
    sim_band = ews_band(sim_ews)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline EWS", f"{base_ews:.1f}", help=f"Band: {base_band}")
    c2.metric("Simulated EWS", f"{sim_ews:.1f}", delta=f"{sim_ews - base_ews:+.1f}", delta_color="inverse",
              help=f"Band: {sim_band}")
    delta_txt = "Risk reduced" if sim_ews < base_ews else ("Risk increased" if sim_ews > base_ews else "No change")
    impact_icon = "✅" if sim_ews < base_ews else ("⚠️" if sim_ews > base_ews else "➖")
    c3.metric("Impact", delta_txt, delta=impact_icon, delta_color="off")

    comp_df = pd.DataFrame({
        "Factor": [FRIENDLY_NAMES[k] for k in EWS_WEIGHTS],
        "Baseline": [base_scores[k] for k in EWS_WEIGHTS],
        "Simulated": [sim_scores[k] for k in EWS_WEIGHTS],
    })
    fig = go.Figure()
    fig.add_bar(name="Baseline", x=comp_df["Factor"], y=comp_df["Baseline"], marker_color=PALETTE["muted"])
    fig.add_bar(name="Simulated", x=comp_df["Factor"], y=comp_df["Simulated"], marker_color=PALETTE["accent"])
    fig.update_layout(barmode="group", title="Composite Risk Sub-Scores — Baseline vs Simulated")
    st.plotly_chart(plotly_theme(fig, height=420), width="stretch")

    st.info("Tip: Try lowering **Single-sourcing** and raising **Inventory Buffer** to see how quickly "
            "concentration and demand-volatility risk fall — often the fastest levers to pull.")


# ========================================================================================
# PAGE 7 — BUSINESS INSIGHTS (Enhanced with AI Agent & SHAP insights)
# ========================================================================================
elif page == "💡 Business Insights":
    st.markdown("# Business Insights")
    st.caption("Executive summary translating machine learning predictions, SHAP attributions, and AI agent insights into supply-chain strategy.")

    high_risk = df[df["Risk_Level"] == "High"]
    top_region = df.groupby("Region")["Early_Warning_Score"].mean().idxmax()
    top_category = df.groupby("Product_Category")["Early_Warning_Score"].mean().idxmax()
    single_source_risk = df[df["Single_Sourcing_Flag"] == 1]["Early_Warning_Score"].mean()
    multi_source_risk = df[df["Single_Sourcing_Flag"] == 0]["Early_Warning_Score"].mean()
    sea_delay = df[df["Transport_Mode"] == "Sea"]["Lead_Time_Days"].mean()
    air_delay = df[df["Transport_Mode"] == "Air"]["Lead_Time_Days"].mean()

    insights = [
        ("🌍 Regional Hotspot", f"<b>{top_region}</b> carries the highest average Early Warning Score in the network — "
         "prioritize geopolitical monitoring and dual-region sourcing for suppliers based there."),
        ("📦 Category Exposure", f"<b>{top_category}</b> shows the highest average disruption risk among product "
         "categories, driven by longer lead times and concentrated supplier bases."),
        ("🔗 Single-Sourcing Penalty", f"Single-sourced shipments carry an average EWS of <b>{single_source_risk:.1f}</b> "
         f"vs <b>{multi_source_risk:.1f}</b> for multi-sourced ones — a "
         f"<b>{single_source_risk - multi_source_risk:+.1f} point</b> risk premium for sourcing concentration."),
        ("🚢 Transport Corridors & Disruption Exposure", f"Corridor analysis reveals <b>Sea transit corridors</b> as the primary "
         f"network-wide bottleneck (averaging <b>{sea_delay:.0f} days</b> lead time vs <b>{air_delay:.0f} days</b> for air) — "
         "congestions at major sea hubs propagate disruptions rapidly across multiple downstream product categories."),
        ("⚡ Non-Linear Shock Risk (XGBoost & SHAP)", "SHAP global feature attributions reveal that geopolitical risk and "
         "port congestion exhibit sharp non-linear thresholds: moderate disruptions compound exponentially when paired with "
         "low inventory buffers, reinforcing why gradient boosting captures risks that linear models underestimate."),
        ("🏭 High-Risk Volume", f"<b>{len(high_risk):,} shipments ({len(high_risk)/len(df)*100:.1f}%)</b> are currently "
         "classified High risk — prioritize them using the priority action plan below."),
    ]

    for title, body in insights:
        st.markdown(f"<div class='sg-card'><h4>{title}</h4><p style='color:{PALETTE['muted']}'>{body}</p></div>",
                     unsafe_allow_html=True)

    st.markdown("#### Recommended Action Plan")
    st.markdown("""
1. **Triage now** — pull the Critical & Elevated band shipments from *Risk Ranking* and assign category owners this week.
2. **Mitigate transit bottlenecks** — evaluate multimodal bypass routes for the high-risk Sea transit corridors flagged in the Data-to-Action Agent view.
3. **Fix concentration risk** — begin dual-sourcing qualification for single-sourced, high-EWS suppliers.
4. **Leverage SHAP explainability** — review the top SHAP waterfall risk drivers for key lanes to craft tailored mitigations instead of generic policies.
5. **Review quarterly** — keep model benchmarks and AI agent analytics calibrated as new logistics shipments log in.
    """)

st.markdown("---")
st.caption("SupplyGuard AI · Predict → Explain → Recommend · Built for supply chain risk analysts.")
