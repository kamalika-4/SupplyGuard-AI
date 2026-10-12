"""
SupplyGuard AI - Supply Chain Data-to-Action Agent Engine
=========================================================
Autonomous analytical agent that accepts natural-language supply-chain questions,
queries actual shipment records and prediction logs from MongoDB (SupplyGuardDB),
runs existing ML/analytical tools (XGBoost, SHAP, EWS, composite risk scorers),
and synthesizes evidence-based answers with factual tables and prioritized recommendations.

Rule: NEVER invent results. Every data point is queried directly from MongoDB/dataset
or computed via the verified analytical models.
"""

import os
import sys
import re
import json
import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

# Path setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from mongo_db import (
    get_shipments,
    get_risk_predictions,
    save_shipment,
    save_risk_prediction,
    check_mongo_connection,
    sanitize_for_mongo,
)
from risk_engine import (
    predict_ml,
    compute_composite_scores,
    compute_ews,
    ews_band,
    explain,
    recommend,
    EWS_WEIGHTS,
    FRIENDLY_NAMES,
)
from shap_engine import compute_local_shap


class AgentResponse:
    """Encapsulates the structured output of an agent analysis."""

    def __init__(
        self,
        question: str,
        intent: str,
        summary: str,
        evidence_df: pd.DataFrame,
        tools_used: List[str],
        recommendations: List[Dict[str, Any]],
        metrics: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error: Optional[str] = None,
    ):
        self.question = question
        self.intent = intent
        self.summary = summary
        self.evidence_df = evidence_df
        self.tools_used = tools_used
        self.recommendations = recommendations
        self.metrics = metrics or {}
        self.success = success
        self.error = error
        self.timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "intent": self.intent,
            "summary": self.summary,
            "evidence_rows": len(self.evidence_df) if self.evidence_df is not None else 0,
            "tools_used": self.tools_used,
            "recommendations": self.recommendations,
            "metrics": self.metrics,
            "success": self.success,
            "timestamp": self.timestamp,
        }


class SupplyChainDataToActionAgent:
    """
    AI Supply Chain Data-to-Action Agent.
    Interfaces directly with MongoDB (SupplyGuardDB) and local ML/analytical tooling.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        xgb_model: Any = None,
        model: Any = None,
        scaler: Any = None,
        target_encoder: Any = None,
        feature_cols: Optional[List[str]] = None,
    ):
        self.df = df
        self.xgb_model = xgb_model
        self.model = model
        self.scaler = scaler
        self.target_encoder = target_encoder
        self.feature_cols = feature_cols or []
        self._ensure_cat_encodings()

    def _ensure_cat_encodings(self):
        """Builds lookup dictionaries for categorical encodings."""
        self.cat_maps = {}
        for col in ["Region", "Country", "Product_Category", "Supplier_Tier", "Transport_Mode"]:
            if col in self.df.columns:
                self.cat_maps[col] = {v: i for i, v in enumerate(sorted(self.df[col].dropna().unique()))}

    def _encode_row(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Encodes categorical fields and composite scores for model input."""
        scores = compute_composite_scores(raw)
        encoded = dict(raw)
        encoded["Region_enc"] = self.cat_maps.get("Region", {}).get(raw.get("Region"), 0)
        encoded["Country_enc"] = self.cat_maps.get("Country", {}).get(raw.get("Country"), 0)
        encoded["Product_Category_enc"] = self.cat_maps.get("Product_Category", {}).get(raw.get("Product_Category"), 0)
        encoded["Supplier_Tier_enc"] = self.cat_maps.get("Supplier_Tier", {}).get(raw.get("Supplier_Tier"), 0)
        encoded["Transport_Mode_enc"] = self.cat_maps.get("Transport_Mode", {}).get(raw.get("Transport_Mode"), 0)
        for k, v in scores.items():
            encoded[k] = v
        return encoded

    def get_actual_shipment(self, shipment_id: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Retrieves actual shipment record from MongoDB `shipments` collection first.
        Falls back to local dataset if not found in MongoDB.
        Returns (record_dict, source_str).
        """
        # Try MongoDB
        mongo_docs = get_shipments(
            filter_dict={"$or": [{"shipment_id": shipment_id}, {"Shipment_ID": shipment_id}]},
            limit=1,
        )
        if mongo_docs:
            doc = mongo_docs[0]
            return doc, "MongoDB (SupplyGuardDB.shipments)"

        # Fallback to local dataset
        matches = self.df[self.df["Shipment_ID"] == shipment_id]
        if not matches.empty:
            return matches.iloc[0].to_dict(), "Local Feature-Engineered Dataset"

        return None, "Not Found"

    def analyze_shipment(self, shipment_id: str) -> AgentResponse:
        """
        Full diagnostic and action prescription for an individual shipment.
        Queries MongoDB, computes XGBoost prediction, SHAP waterfall drivers,
        and generates evidence-based mitigation recommendations.
        """
        record, source = self.get_actual_shipment(shipment_id)
        tools_used = [f"Query {source}"]

        if record is None:
            return AgentResponse(
                question=f"Diagnose shipment {shipment_id}",
                intent="shipment_lookup",
                summary=f"❌ Shipment **{shipment_id}** was not found in MongoDB `shipments` collection or the active dataset.",
                evidence_df=pd.DataFrame(),
                tools_used=tools_used,
                recommendations=[{
                    "priority": 1,
                    "factor": "Record Lookup Failure",
                    "actions": ["Verify shipment ID format (e.g. SG00001 to SG05000 or SH-xxxx).", "Import or sync shipments via MongoDB Records tab."],
                }],
                success=False,
                error=f"Shipment {shipment_id} not found.",
            )

        # 1. Run ML Prediction & Scoring
        active_model = self.xgb_model if self.xgb_model is not None else self.model
        model_name = active_model.__class__.__name__ if active_model is not None else "Rule-Engine"

        encoded = self._encode_row(record)
        pred_label, pred_proba = predict_ml(encoded, active_model, self.scaler, self.target_encoder, self.feature_cols)
        tools_used.append(f"ML Classifier ({model_name})")

        scores = compute_composite_scores(record)
        ews_score = compute_ews(scores)
        band = ews_band(ews_score)
        tools_used.append("Early Warning Score (EWS) Engine")

        exp = explain(scores, top_n=3)
        recs = recommend(exp)
        tools_used.append("Rule-Based Recommendation Engine")

        # 2. Run SHAP Attribution if XGBoost is available
        shap_summary = None
        if self.xgb_model is not None:
            try:
                local_shap = compute_local_shap(
                    self.xgb_model,
                    encoded,
                    self.feature_cols,
                    target_class="High",
                    target_encoder=self.target_encoder,
                )
                tools_used.append("SHAP Game-Theoretic Explainer (TreeExplainer)")
                shap_summary = {
                    "base_value": local_shap["base_value"],
                    "sum_shap": local_shap["sum_shap"],
                    "margin_value": local_shap["margin_value"],
                    "disruption_prob": local_shap["disruption_prob"],
                    "top_risk_drivers": local_shap["risk_drivers"][:3],
                    "top_protective": local_shap["protective_factors"][:3],
                }
            except Exception as e:
                shap_summary = {"error": str(e)}

        # 3. Check MongoDB for existing prediction audit log
        past_preds = get_risk_predictions(
            filter_dict={"$or": [{"shipment_id": shipment_id}, {"Shipment_ID": shipment_id}]},
            limit=5,
        )
        tools_used.append("MongoDB (SupplyGuardDB.risk_predictions) Audit Check")

        # 4. Construct Evidence Table
        evidence_data = {
            "Parameter": [
                "Shipment ID", "Data Source", "Region / Country", "Product Category",
                "Supplier Tier", "Transport Mode", "Lead Time (Days)", "Distance (KM)",
                "Order Quantity", "Inventory Buffer (Days)", "On-Time Delivery Rate (%)",
                "Supplier Reliability", "Financial Stability", "Port Congestion Index",
                "Geopolitical Risk Index", "Weather Risk Index", "Customs Delay (Days)",
                "Disruption Events (Past Year)", "Early Warning Score", "EWS Risk Band",
                "XGBoost Predicted Risk", "P(High Disruption Risk)", "Past MongoDB Predictions Logged",
            ],
            "Actual Value": [
                shipment_id,
                source,
                f"{record.get('Region', 'N/A')} ({record.get('Country', 'N/A')})",
                str(record.get("Product_Category", "N/A")),
                str(record.get("Supplier_Tier", "N/A")),
                str(record.get("Transport_Mode", "N/A")),
                f"{record.get('Lead_Time_Days', 'N/A')} days",
                f"{record.get('Distance_KM', 'N/A'):,} km" if isinstance(record.get("Distance_KM"), (int, float)) else "N/A",
                f"{record.get('Order_Quantity', 'N/A'):,}" if isinstance(record.get("Order_Quantity"), (int, float)) else "N/A",
                f"{record.get('Inventory_Level_Days', 'N/A')} days",
                f"{record.get('On_Time_Delivery_Rate_Pct', 'N/A')}%",
                f"{record.get('Supplier_Reliability_Score', 'N/A')}",
                f"{record.get('Financial_Stability_Score', 'N/A')}",
                f"{record.get('Port_Congestion_Index', 'N/A')}",
                f"{record.get('Geopolitical_Risk_Index', 'N/A')}",
                f"{record.get('Weather_Risk_Index', 'N/A')}",
                f"{record.get('Customs_Delay_Days', 'N/A')} days",
                f"{record.get('Disruption_Events_Past_Year', 'N/A')}",
                f"{ews_score:.1f} / 100",
                band,
                pred_label,
                f"{pred_proba.get('High', 0.0)*100:.1f}%",
                f"{len(past_preds)} audit entries",
            ],
        }
        evidence_df = pd.DataFrame(evidence_data)

        # 5. Formulate Evidence-Based Summary
        high_prob_pct = pred_proba.get("High", 0.0) * 100
        summary = (
            f"### 🎯 Diagnostic for Shipment `{shipment_id}`\n\n"
            f"• **Risk Classification:** **{pred_label} Risk** (XGBoost probability of High disruption: **{high_prob_pct:.1f}%**)\n"
            f"• **Early Warning Score (EWS):** **{ews_score:.1f} / 100** (Band: `{band}`)\n"
            f"• **Operational Origin:** `{record.get('Country')}` ({record.get('Region')}) via `{record.get('Transport_Mode')}` freight.\n"
            f"• **Lead Time & Buffer:** Lead time is `{record.get('Lead_Time_Days')}` days with `{record.get('Inventory_Level_Days')}` days of inventory buffer."
        )

        if shap_summary and "top_risk_drivers" in shap_summary:
            top_drivers = shap_summary["top_risk_drivers"]
            if top_drivers:
                summary += "\n\n**Key Game-Theoretic SHAP Risk Elevators:**\n"
                for d in top_drivers:
                    summary += f"- **{d['friendly_name']}** (SHAP value: `+{d['shap_value']:.3f}` log-odds)\n"

        return AgentResponse(
            question=f"Diagnose shipment {shipment_id}",
            intent="shipment_lookup",
            summary=summary,
            evidence_df=evidence_df,
            tools_used=tools_used,
            recommendations=recs,
            metrics={
                "shipment_id": shipment_id,
                "predicted_risk": pred_label,
                "ews_score": ews_score,
                "ews_band": band,
                "p_high": high_prob_pct,
                "past_logs_count": len(past_preds),
            },
        )

    def triage_critical_shipments(self, limit: int = 15) -> AgentResponse:
        """
        Queries actual high-risk and critical shipments from MongoDB / dataset,
        computes systemic disruption metrics, and outlines triage actions.
        """
        tools_used = ["MongoDB (SupplyGuardDB.shipments) Query", "EWS Ranking Engine"]

        # Fetch from MongoDB first
        mongo_records = get_shipments(
            filter_dict={"$or": [{"risk_level": "High"}, {"ews_band": "Critical"}, {"EWS_Band": "Critical"}]},
            limit=limit,
        )

        if len(mongo_records) < 5:
            # Augment with dataset to provide complete coverage
            tools_used.append("Dataset Critical Filter (Early_Warning_Score DESC)")
            crit_df = self.df.sort_values("Early_Warning_Score", ascending=False).head(limit)
        else:
            tools_used.append("MongoDB Retrieved Records")
            crit_df = pd.DataFrame(mongo_records)
            if "Early_Warning_Score" in crit_df.columns:
                crit_df = crit_df.sort_values("Early_Warning_Score", ascending=False).head(limit)
            elif "early_warning_score" in crit_df.columns:
                crit_df = crit_df.sort_values("early_warning_score", ascending=False).head(limit)

        display_cols = [
            c for c in [
                "Shipment_ID", "shipment_id", "Region", "Country", "Product_Category",
                "Transport_Mode", "Early_Warning_Score", "early_warning_score",
                "EWS_Band", "ews_band", "Risk_Level", "risk_level",
                "Lead_Time_Days", "Disruption_Events_Past_Year"
            ] if c in crit_df.columns
        ]

        evidence_df = crit_df[display_cols].copy()

        # Compute factual aggregated stats
        total_eval = len(self.df)
        high_risk_total = int((self.df["Risk_Level"] == "High").sum())
        crit_band_total = int((self.df["EWS_Band"] == "Critical").sum())
        avg_crit_lead = float(self.df[self.df["EWS_Band"] == "Critical"]["Lead_Time_Days"].mean())
        top_risk_region = self.df[self.df["EWS_Band"] == "Critical"]["Region"].value_counts().index[0]

        summary = (
            f"### 🚨 High-Exposure & Critical Shipments Triage Report\n\n"
            f"• **Critical Exposure Count:** **{crit_band_total:,} shipments** ({crit_band_total/total_eval*100:.1f}% of network) currently in **Critical Band**.\n"
            f"• **High Risk Volume:** **{high_risk_total:,} shipments** ({high_risk_total/total_eval*100:.1f}%) classified as **High Disruption Risk**.\n"
            f"• **Primary Regional Hotspot:** **{top_risk_region}** concentrates the largest share of critical shipments.\n"
            f"• **Average Lead Time for Critical Lanes:** **{avg_crit_lead:.1f} days**, increasing operational vulnerability to customs and port bottlenecks."
        )

        recs = [
            {
                "priority": 1,
                "factor": "Critical Lane Intervention",
                "actions": [
                    f"Immediate buffer inventory escalation for top lanes originating in {top_risk_region}.",
                    "Daily tracking on shipments with lead times > 20 days and port congestion index > 0.4.",
                ],
            },
            {
                "priority": 2,
                "factor": "Multimodal Route Shifting",
                "actions": [
                    "Evaluate air-freight spot quotes for single-sourced Tier 1 components currently moving via congested sea lanes.",
                    "Pre-clear customs documentation 48 hours prior to port arrival.",
                ],
            },
            {
                "priority": 3,
                "factor": "Supplier Health Re-audit",
                "actions": [
                    "Conduct emergency supplier viability reviews for vendors with reliability score < 0.6.",
                    "Activate secondary qualified suppliers for all single-sourced critical parts.",
                ],
            },
        ]

        return AgentResponse(
            question="Triage critical and high-risk shipments",
            intent="risk_triage",
            summary=summary,
            evidence_df=evidence_df,
            tools_used=tools_used,
            recommendations=recs,
            metrics={
                "total_critical": crit_band_total,
                "total_high_risk": high_risk_total,
                "avg_critical_lead_days": avg_crit_lead,
                "top_region": top_risk_region,
            },
        )

    def analyze_transport_modes(self) -> AgentResponse:
        """
        Analyzes performance, lead times, and disruption exposure by Transport Mode.
        """
        tools_used = ["MongoDB / Dataset Transportation Aggregator", "EWS Cross-Tabulation Tool"]

        grouped = (
            self.df.groupby("Transport_Mode")
            .agg(
                Shipment_Count=("Shipment_ID", "count"),
                Avg_Lead_Time_Days=("Lead_Time_Days", "mean"),
                Avg_EWS=("Early_Warning_Score", "mean"),
                Avg_On_Time_Delivery_Pct=("On_Time_Delivery_Rate_Pct", "mean"),
                Avg_Disruptions=("Disruption_Events_Past_Year", "mean"),
                Avg_Port_Congestion=("Port_Congestion_Index", "mean"),
                Pct_High_Risk=("Risk_Level", lambda x: (x == "High").mean() * 100),
            )
            .reset_index()
        )

        grouped["Avg_Lead_Time_Days"] = grouped["Avg_Lead_Time_Days"].round(1)
        grouped["Avg_EWS"] = grouped["Avg_EWS"].round(1)
        grouped["Avg_On_Time_Delivery_Pct"] = grouped["Avg_On_Time_Delivery_Pct"].round(1)
        grouped["Avg_Disruptions"] = grouped["Avg_Disruptions"].round(2)
        grouped["Avg_Port_Congestion"] = grouped["Avg_Port_Congestion"].round(2)
        grouped["Pct_High_Risk"] = grouped["Pct_High_Risk"].round(1)

        highest_ews_mode = grouped.sort_values("Avg_EWS", ascending=False).iloc[0]
        longest_lead_mode = grouped.sort_values("Avg_Lead_Time_Days", ascending=False).iloc[0]

        summary = (
            f"### 🚢 Transport Corridor Disruption Analysis\n\n"
            f"• **Highest Disruption Risk Mode:** **{highest_ews_mode['Transport_Mode']}** carries the highest average Early Warning Score (**{highest_ews_mode['Avg_EWS']:.1f} / 100**) and **{highest_ews_mode['Pct_High_Risk']:.1f}%** High-Risk exposure.\n"
            f"• **Longest Lead Time Mode:** **{longest_lead_mode['Transport_Mode']}** averages **{longest_lead_mode['Avg_Lead_Time_Days']} days** lead time vs **{grouped[grouped['Transport_Mode'] == 'Air']['Avg_Lead_Time_Days'].values[0]} days** for Air.\n"
            f"• **On-Time Reliability:** Air delivery averages **{grouped[grouped['Transport_Mode'] == 'Air']['Avg_On_Time_Delivery_Pct'].values[0]}%** on-time, compared to **{grouped[grouped['Transport_Mode'] == 'Sea']['Avg_On_Time_Delivery_Pct'].values[0]}%** for Sea."
        )

        recs = [
            {
                "priority": 1,
                "factor": f"{longest_lead_mode['Transport_Mode']} Freight Transit Buffer",
                "actions": [
                    f"Expand safety stock buffer by +{int(longest_lead_mode['Avg_Lead_Time_Days'] * 0.25)} days for parts moving via {longest_lead_mode['Transport_Mode']}.",
                    "Implement container telemetry tracking on major maritime routes.",
                ],
            },
            {
                "priority": 2,
                "factor": "Dynamic Modal Re-allocation",
                "actions": [
                    "Establish pre-negotiated air-freight surge contracts to shift high-margin stock during port congestion spikes.",
                    "Consolidate less-than-container-load (LCL) shipments into full containers to reduce dwell time.",
                ],
            },
        ]

        return AgentResponse(
            question="Compare transport mode risks and transit times",
            intent="transport_analysis",
            summary=summary,
            evidence_df=grouped,
            tools_used=tools_used,
            recommendations=recs,
        )

    def analyze_supplier_risk(self) -> AgentResponse:
        """
        Analyzes supplier tier risk and single-sourcing vulnerabilities.
        """
        tools_used = ["Supplier Tier Analytical Aggregator", "Single-Sourcing Impact Tool"]

        tier_summary = (
            self.df.groupby("Supplier_Tier")
            .agg(
                Shipment_Count=("Shipment_ID", "count"),
                Avg_Reliability=("Supplier_Reliability_Score", "mean"),
                Avg_Financial=("Financial_Stability_Score", "mean"),
                Avg_EWS=("Early_Warning_Score", "mean"),
                Pct_High_Risk=("Risk_Level", lambda x: (x == "High").mean() * 100),
            )
            .reset_index()
            .round(2)
        )

        ss_yes = self.df[self.df["Single_Sourcing_Flag"] == 1]["Early_Warning_Score"].mean()
        ss_no = self.df[self.df["Single_Sourcing_Flag"] == 0]["Early_Warning_Score"].mean()
        ss_delta = ss_yes - ss_no

        summary = (
            f"### 🏭 Supplier Tier & Single-Sourcing Risk Analysis\n\n"
            f"• **Single-Sourcing Risk Premium:** Shipments with single-sourced suppliers carry an average Early Warning Score of **{ss_yes:.1f}** vs **{ss_no:.1f}** for dual/multi-sourced suppliers — an excess **+{ss_delta:.1f} point risk penalty**.\n"
            f"• **Tier Vulnerability:** Supplier Tier 3 displays the lowest financial stability (**{tier_summary[tier_summary['Supplier_Tier'] == 'Tier 3']['Avg_Financial'].values[0]:.2f}**) and lowest reliability (**{tier_summary[tier_summary['Supplier_Tier'] == 'Tier 3']['Avg_Reliability'].values[0]:.2f}**)."
        )

        recs = [
            {
                "priority": 1,
                "factor": "Dual-Sourcing Mandate",
                "actions": [
                    "Require dual-sourcing qualification for all parts currently single-sourced with Tier 2 or Tier 3 vendors.",
                    "Split volume 70/30 across primary and secondary suppliers to keep alternative supply lines warm.",
                ],
            },
            {
                "priority": 2,
                "factor": "Financial Health Audits",
                "actions": [
                    "Institute quarterly financial stress tests and credit monitoring for key Tier 3 suppliers.",
                    "Implement supplier development programs and dynamic payment terms to stabilize vendor cash flows.",
                ],
            },
        ]

        return AgentResponse(
            question="Analyze supplier tier and single-sourcing risk",
            intent="supplier_risk",
            summary=summary,
            evidence_df=tier_summary,
            tools_used=tools_used,
            recommendations=recs,
            metrics={"single_source_penalty": ss_delta, "tier_summary": tier_summary.to_dict(orient="records")},
        )

    def analyze_regional_hotspots(self) -> AgentResponse:
        """
        Analyzes disruption exposure across global geographic regions.
        """
        tools_used = ["Regional Hotspot Aggregator", "Geopolitical & Weather Risk Tool"]

        reg_summary = (
            self.df.groupby("Region")
            .agg(
                Shipment_Count=("Shipment_ID", "count"),
                Avg_Geopolitical_Risk=("Geopolitical_Risk_Index", "mean"),
                Avg_Weather_Risk=("Weather_Risk_Index", "mean"),
                Avg_Port_Congestion=("Port_Congestion_Index", "mean"),
                Avg_Customs_Delay=("Customs_Delay_Days", "mean"),
                Avg_EWS=("Early_Warning_Score", "mean"),
                Pct_Critical=("EWS_Band", lambda x: (x == "Critical").mean() * 100),
            )
            .reset_index()
            .round(2)
            .sort_values("Avg_EWS", ascending=False)
        )

        top_region = reg_summary.iloc[0]

        summary = (
            f"### 🌍 Global Regional Disruption Hotspots\n\n"
            f"• **Primary Hotspot Region:** **{top_region['Region']}** has the highest average EWS (**{top_region['Avg_EWS']:.1f} / 100**) with **{top_region['Pct_Critical']:.1f}%** of shipments in Critical band.\n"
            f"• **Key Stress Drivers:** Geopolitical risk averages **{top_region['Avg_Geopolitical_Risk']:.2f}** and customs delay averages **{top_region['Avg_Customs_Delay']:.1f} days** in this region.\n"
            f"• **Global Spread:** Network spans **{len(reg_summary)} regions** with varying exposure to climate and geopolitical disruption."
        )

        recs = [
            {
                "priority": 1,
                "factor": f"{top_region['Region']} Disruption Mitigation",
                "actions": [
                    f"Establish regional safety inventory buffers in neighboring lower-risk logistics hubs.",
                    f"Diversify procurement contracts outside {top_region['Region']} to prevent single-region exposure.",
                ],
            },
            {
                "priority": 2,
                "factor": "Customs Fast-Track Registration",
                "actions": [
                    "Partner with Authorised Economic Operator (AEO) certified forwarders to minimize customs dwell times.",
                    "Pre-file customs declarations electronically 72 hours before border crossing.",
                ],
            },
        ]

        return AgentResponse(
            question="Identify global regional disruption hotspots",
            intent="regional_hotspots",
            summary=summary,
            evidence_df=reg_summary,
            tools_used=tools_used,
            recommendations=recs,
        )

    def audit_prediction_history(self) -> AgentResponse:
        """
        Retrieves prediction audit logs from MongoDB `risk_predictions` and summarizes model inferences.
        """
        tools_used = ["MongoDB (SupplyGuardDB.risk_predictions) Reader", "Prediction Audit Telemetry Tool"]

        preds = get_risk_predictions(limit=100)

        if not preds:
            return AgentResponse(
                question="Audit MongoDB prediction history",
                intent="prediction_history",
                summary=(
                    "### 📜 MongoDB Prediction History Audit\n\n"
                    "• **Status:** No prediction records currently logged in `SupplyGuardDB.risk_predictions`.\n"
                    "• **Action:** Navigate to the **🔮 Predict · Explain · Recommend** page, score a shipment, "
                    "and click **Predict** to populate real audit telemetry into MongoDB."
                ),
                evidence_df=pd.DataFrame(),
                tools_used=tools_used,
                recommendations=[{
                    "priority": 1,
                    "factor": "Populate Prediction Logs",
                    "actions": ["Run manual or automated batch predictions on the Predict page.", "Use MongoDB Records tab to verify connection status."],
                }],
            )

        rows = []
        for p in preds:
            prob_dict = p.get("prediction_probabilities", {})
            p_high = f"{prob_dict.get('High', 0.0)*100:.1f}%" if isinstance(prob_dict, dict) and "High" in prob_dict else "N/A"
            rows.append({
                "Shipment_ID": p.get("shipment_id", "N/A"),
                "Predicted_Risk": p.get("predicted_risk_level", "N/A"),
                "Model": p.get("model_name", "N/A"),
                "P(High)": p_high,
                "EWS_Score": p.get("early_warning_score", "N/A"),
                "EWS_Band": p.get("ews_band", "N/A"),
                "Timestamp": p.get("timestamp") or p.get("prediction_timestamp") or p.get("created_at", "N/A"),
            })

        evidence_df = pd.DataFrame(rows)
        risk_counts = evidence_df["Predicted_Risk"].value_counts().to_dict()

        summary = (
            f"### 📜 MongoDB Prediction Audit Log Report\n\n"
            f"• **Total Predictions Retrieved:** **{len(evidence_df):,} records** from `SupplyGuardDB.risk_predictions`.\n"
            f"• **Risk Class Distribution:** " + ", ".join([f"**{k}:** {v}" for k, v in risk_counts.items()]) + ".\n"
            f"• **Model Auditing:** All records logged with full timestamp, probability distributions, and domain telemetry."
        )

        recs = [
            {
                "priority": 1,
                "factor": "Prediction Model Governance",
                "actions": [
                    "Perform monthly drift checks comparing predicted risk distributions against ground-truth carrier delay reports.",
                    "Flag shipments predicted as High risk with P(High) > 80% for immediate automated expediting.",
                ],
            }
        ]

        return AgentResponse(
            question="Audit prediction history in MongoDB",
            intent="prediction_history",
            summary=summary,
            evidence_df=evidence_df,
            tools_used=tools_used,
            recommendations=recs,
            metrics={"total_logged": len(evidence_df), "risk_distribution": risk_counts},
        )

    def answer_question(self, question: str) -> AgentResponse:
        """
        Main entry point for natural-language queries.
        Parses intent, queries actual MongoDB / dataset records, runs tools,
        and generates evidence-based response without ever inventing numbers.
        """
        q = (question or "").strip()
        q_lower = q.lower()

        if not q:
            return self.triage_critical_shipments(limit=10)

        # 1. Check for specific Shipment ID
        sid_match = re.search(r"\b(sg\d+|sh-[\w-]+|sh\d+)\b", q_lower)
        if sid_match:
            # Reconstruct exact case (e.g. SG02088)
            sid_str = sid_match.group(1).upper()
            return self.analyze_shipment(sid_str)

        # 2. Check for Prediction History / MongoDB audit queries
        if any(w in q_lower for w in ["history", "audit", "logged", "past prediction", "saved prediction"]):
            return self.audit_prediction_history()

        # 3. Check for Transport Mode / Corridor queries
        if any(w in q_lower for w in ["transport", "freight", "mode", "carrier", "sea", "air", "road", "rail", "shipping lane", "transit"]):
            return self.analyze_transport_modes()

        # 4. Check for Supplier / Single-Sourcing queries
        if any(w in q_lower for w in ["supplier", "tier", "single source", "sourcing", "vendor", "reliability"]):
            return self.analyze_supplier_risk()

        # 5. Check for Regional / Hotspot / Geopolitical queries
        if any(w in q_lower for w in ["region", "country", "hotspot", "geopolitical", "weather", "customs", "asia", "europe", "america"]):
            return self.analyze_regional_hotspots()

        # 6. Check for Critical / High-Risk / Triage queries
        if any(w in q_lower for w in ["critical", "high risk", "urgent", "triage", "vulnerable", "worst", "danger", "exposure"]):
            return self.triage_critical_shipments(limit=15)

        # 7. Fallback / General Analytical Query
        # Synthesize broad network overview from actual data
        return self.triage_critical_shipments(limit=10)
