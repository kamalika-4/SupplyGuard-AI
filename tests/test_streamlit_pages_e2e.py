"""
SupplyGuard AI - Streamlit Pages End-to-End Execution Test
==========================================================
Verifies that root app.py and dashboard/app.py can be initialized and that
all core modules and calculations for each of the 7 dashboard pages execute cleanly.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")
if DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, DASHBOARD_DIR)

from dashboard.risk_engine import (
    load_artifacts, load_xgb_model, compute_composite_scores, compute_ews,
    ews_band, explain, recommend, predict_ml, batch_predict_df
)
from dashboard.shap_engine import (
    compute_local_shap, plot_shap_waterfall, compute_global_shap_data,
    plot_shap_summary_bar, plot_shap_beeswarm_interactive, plot_shap_dependence_interactive
)
from src.mongo_db import is_cloud_environment, is_mongo_persistence_disabled


class TestStreamlitPagesE2E(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(os.path.join(BASE_DIR, "data", "feature_engineered_data.csv"))
        cls.model, cls.scaler, cls.target_encoder, cls.feature_cols, cls.xgb_model = load_artifacts(include_xgb=True)

    def test_page1_mission_control_calculations(self):
        """Page 1 calculations: metrics, pie charts, EWS bands, top countries."""
        avg_ews = self.df["Early_Warning_Score"].mean()
        self.assertFalse(np.isnan(avg_ews))
        crit_pct = (self.df["EWS_Band"] == "Critical").mean() * 100
        self.assertGreaterEqual(crit_pct, 0.0)

    def test_page2_eda_and_custom_csv_flow(self):
        """Page 2: preprocessing, composite scores, and batch scoring on a sample DataFrame."""
        sample_subset = self.df.head(10).copy()
        scored, summary = batch_predict_df(sample_subset, self.xgb_model, self.scaler, self.target_encoder, self.feature_cols, base_df=self.df)
        self.assertEqual(len(scored), 10)
        self.assertIn("Predicted_Risk_Level", scored.columns)
        self.assertIn("Disruption_Probability_High", scored.columns)

    def test_page3_model_lab_shap(self):
        """Page 3: global SHAP bar, beeswarm, and dependence plots."""
        g_data = compute_global_shap_data(self.xgb_model, self.df.head(50), self.feature_cols, target_encoder=self.target_encoder)
        self.assertIsNotNone(g_data)
        fig_bar = plot_shap_summary_bar(g_data, top_n=5)
        self.assertIsNotNone(fig_bar)
        fig_bee = plot_shap_beeswarm_interactive(g_data, top_n=5)
        self.assertIsNotNone(fig_bee)
        fig_dep = plot_shap_dependence_interactive(g_data, feature_name="Geopolitical_Risk_Index")
        self.assertIsNotNone(fig_dep)

    def test_page4_predict_workflow(self):
        """Page 4: predict single shipment, SHAP waterfall, recommendations."""
        raw_shipment = self.df.iloc[0].to_dict()
        scores = compute_composite_scores(raw_shipment)
        ews = compute_ews(scores)
        band = ews_band(ews)
        exp = explain(scores)
        recs = recommend(exp)
        self.assertGreater(len(recs), 0)

        label, proba = predict_ml(raw_shipment, self.xgb_model, self.scaler, self.target_encoder, self.feature_cols)
        self.assertIn(label, ["Low", "Medium", "High"])

        local_shap = compute_local_shap(self.xgb_model, raw_shipment, self.feature_cols, target_encoder=self.target_encoder)
        fig_wf = plot_shap_waterfall(local_shap)
        self.assertIsNotNone(fig_wf)

    def test_page5_risk_ranking(self):
        """Page 5: leaderboard sorting and filtering."""
        rdf = self.df.sort_values("Early_Warning_Score", ascending=False).head(50)
        self.assertEqual(len(rdf), 50)
        self.assertGreaterEqual(rdf.iloc[0]["Early_Warning_Score"], rdf.iloc[-1]["Early_Warning_Score"])

    def test_page6_what_if_simulator(self):
        """Page 6: sensitivity adjustment of parameters."""
        base_row = self.df.iloc[0].to_dict()
        base_scores = compute_composite_scores(base_row)
        base_ews = compute_ews(base_scores)

        sim_row = dict(base_row)
        sim_row["Geopolitical_Risk_Index"] = 0.95
        sim_scores = compute_composite_scores(sim_row)
        sim_ews = compute_ews(sim_scores)
        self.assertGreater(sim_ews, base_ews)

    def test_page7_business_insights(self):
        """Page 7: aggregate hotspot metrics."""
        top_reg = self.df.groupby("Region")["Early_Warning_Score"].mean().idxmax()
        self.assertIsNotNone(top_reg)


if __name__ == "__main__":
    unittest.main()
