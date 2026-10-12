"""
SupplyGuard AI - End-to-End Test Suite
======================================
Tests all enhanced features: XGBoost classifier, SHAP explainable AI,
AI Supply Chain Data-to-Action Agent, Risk Engine, Preprocessing, and Dashboard integrity.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import joblib

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "dashboard"))

from dashboard.risk_engine import (
    load_artifacts, load_xgb_model, compute_composite_scores,
    compute_ews, ews_band, explain, recommend, predict_ml
)
from dashboard.agent_engine import SupplyChainDataToActionAgent
from src.mongo_db import get_risk_predictions, save_risk_prediction
from dashboard.shap_engine import (
    compute_local_shap, plot_shap_waterfall,
    compute_global_shap_data, plot_shap_summary_bar,
    plot_shap_beeswarm_interactive, plot_shap_dependence_interactive
)
from src.preprocessing import load_raw, preprocess


class TestSupplyGuardAI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_path = os.path.join(BASE_DIR, "data", "feature_engineered_data.csv")
        cls.df = pd.read_csv(cls.data_path)
        cls.model, cls.scaler, cls.target_encoder, cls.feature_cols = load_artifacts()
        cls.xgb_model = load_xgb_model()

    def test_01_dataset_integrity(self):
        """Verifies dataset has 5,000 rows and required columns."""
        self.assertEqual(len(self.df), 5000)
        self.assertIn("Shipment_ID", self.df.columns)
        self.assertIn("Risk_Level", self.df.columns)
        self.assertIn("Early_Warning_Score", self.df.columns)
        self.assertIn("EWS_Band", self.df.columns)
        for col in self.feature_cols:
            self.assertIn(col, self.df.columns)

    def test_02_xgboost_classifier(self):
        """Verifies XGBoost model loads, predicts valid classes, and produces probabilities."""
        self.assertIsNotNone(self.xgb_model, "XGBoost model file missing or not loaded")
        
        sample = self.df.iloc[0].to_dict()
        label, proba = predict_ml(
            sample, self.xgb_model, self.scaler, self.target_encoder, self.feature_cols
        )
        self.assertIn(label, ["Low", "Medium", "High"])
        self.assertAlmostEqual(sum(proba.values()), 1.0, places=2)
        self.assertEqual(len(proba), 3)

    def test_03_shap_local_waterfall(self):
        """Verifies local SHAP calculation, exact additivity, probabilities, and waterfall generation across shipments."""
        sample_0 = self.df.iloc[0].to_dict()
        local_shap_0 = compute_local_shap(
            self.xgb_model, sample_0, self.feature_cols, target_class="High", target_encoder=self.target_encoder
        )
        self.assertIn("feature_shap", local_shap_0)
        self.assertIn("base_value", local_shap_0)
        self.assertIn("margin_value", local_shap_0)
        self.assertIn("sum_shap", local_shap_0)
        self.assertIn("disruption_prob", local_shap_0)
        self.assertIn("baseline_prob", local_shap_0)
        self.assertIn("risk_drivers", local_shap_0)
        self.assertIn("protective_factors", local_shap_0)
        self.assertEqual(len(local_shap_0["feature_shap"]), len(self.feature_cols))

        # 1. Exact mathematical additivity: Base + Sum(SHAP) == Margin
        additivity_diff = abs((local_shap_0["base_value"] + local_shap_0["sum_shap"]) - local_shap_0["margin_value"])
        self.assertLess(additivity_diff, 1e-4, f"Additivity violated: diff={additivity_diff}")

        # 2. Valid probability bounds
        self.assertGreaterEqual(local_shap_0["disruption_prob"], 0.0)
        self.assertLessEqual(local_shap_0["disruption_prob"], 1.0)
        self.assertGreaterEqual(local_shap_0["baseline_prob"], 0.0)
        self.assertLessEqual(local_shap_0["baseline_prob"], 1.0)

        # 3. Dynamic shipment test: compare with a different shipment
        sample_1 = self.df.iloc[10].to_dict()
        local_shap_1 = compute_local_shap(
            self.xgb_model, sample_1, self.feature_cols, target_class="High", target_encoder=self.target_encoder
        )
        # Verify base values match (same model) but shipments have distinct SHAP values
        self.assertAlmostEqual(local_shap_0["base_value"], local_shap_1["base_value"], places=4)
        self.assertNotEqual(local_shap_0["margin_value"], local_shap_1["margin_value"])
        self.assertNotEqual(local_shap_0["sum_shap"], local_shap_1["sum_shap"])

        # 4. Waterfall plot generation
        fig_wf_0 = plot_shap_waterfall(local_shap_0)
        self.assertGreater(len(fig_wf_0.data), 0)
        fig_wf_1 = plot_shap_waterfall(local_shap_1)
        self.assertGreater(len(fig_wf_1.data), 0)

    def test_04_shap_global_visualizations(self):
        """Verifies global SHAP computation, summary bar, beeswarm, and dependence plots."""
        global_shap = compute_global_shap_data(
            self.xgb_model, self.df, self.feature_cols, self.target_encoder, sample_size=50
        )
        self.assertIn("contribs", global_shap)
        self.assertEqual(global_shap["contribs"].shape[0], 50)

        fig_bar = plot_shap_summary_bar(global_shap, top_n=10)
        self.assertEqual(len(fig_bar.data), 3)  # 3 risk classes

        fig_bee = plot_shap_beeswarm_interactive(global_shap, target_class="High", top_n=10)
        self.assertGreater(len(fig_bee.data), 0)

        fig_dep = plot_shap_dependence_interactive(global_shap, feature_name="Geopolitical_Risk_Index")
        self.assertGreater(len(fig_dep.data), 0)

    def test_05_data_to_action_agent_and_prediction_history(self):
        """Verifies AI Data-to-Action Agent natural-language query processing,
        MongoDB querying, ML tool execution, evidence table generation, and prediction history deduplication."""
        # 1. Initialize agent
        agent = SupplyChainDataToActionAgent(
            df=self.df,
            xgb_model=self.xgb_model,
            model=self.model,
            scaler=self.scaler,
            target_encoder=self.target_encoder,
            feature_cols=self.feature_cols,
        )

        # 2. Test natural-language shipment inquiry
        sample_sid = self.df["Shipment_ID"].iloc[0]
        res_shipment = agent.answer_question(f"What is the disruption risk for shipment {sample_sid}?")
        self.assertTrue(res_shipment.success)
        self.assertEqual(res_shipment.intent, "shipment_lookup")
        self.assertGreater(len(res_shipment.evidence_df), 0)
        self.assertGreater(len(res_shipment.recommendations), 0)
        self.assertIn("Shipment ID", res_shipment.evidence_df["Parameter"].values)
        self.assertIn("Early Warning Score", res_shipment.evidence_df["Parameter"].values)
        self.assertIn("XGBoost Predicted Risk", res_shipment.evidence_df["Parameter"].values)

        # 3. Test critical triage natural-language question
        res_triage = agent.answer_question("Triage critical and high-risk shipments across the network")
        self.assertTrue(res_triage.success)
        self.assertEqual(res_triage.intent, "risk_triage")
        self.assertGreater(len(res_triage.evidence_df), 0)
        self.assertIn("Early_Warning_Score", res_triage.evidence_df.columns)
        self.assertGreater(len(res_triage.recommendations), 0)

        # 4. Test transport corridor inquiry
        res_transport = agent.answer_question("Compare transport modes Sea vs Air and find bottlenecks")
        self.assertTrue(res_transport.success)
        self.assertEqual(res_transport.intent, "transport_analysis")
        self.assertGreater(len(res_transport.evidence_df), 0)
        self.assertIn("Transport_Mode", res_transport.evidence_df.columns)

        # 5. Test Prediction History saving with duplicate prevention
        test_payload = {
            "shipment_id": f"TEST-DEDUP-{sample_sid}",
            "model_name": "XGBoostClassifier",
            "predicted_risk_level": "Medium",
            "risk_probability": 0.45,
            "prediction_probabilities": {"Low": 0.35, "Medium": 0.45, "High": 0.20},
            "early_warning_score": 52.3,
            "ews_band": "Watch",
            "shap_summary": {"base_value": 0.1, "margin_value": 0.3, "top_risk_drivers": []},
            "shipment_inputs": {"Region": "East Asia", "Lead_Time_Days": 14},
            "submission_hash": f"hash_test_{sample_sid}",
        }
        # First save: should succeed
        save_1 = save_risk_prediction(test_payload, prevent_duplicates=True)
        self.assertTrue(save_1.get("success"))

        # Duplicate save with same submission_hash: should prevent duplicate
        save_2 = save_risk_prediction(test_payload, prevent_duplicates=True)
        self.assertTrue(save_2.get("success"))
        self.assertTrue(save_2.get("duplicate"))
        self.assertEqual(save_1.get("prediction_id"), save_2.get("prediction_id"))

        # Verify prediction history retrieval
        preds = get_risk_predictions(filter_dict={"submission_hash": f"hash_test_{sample_sid}"})
        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0].get("predicted_risk_level"), "Medium")

    def test_06_risk_engine_logic(self):
        """Verifies Early Warning Score, Bands, Explanation, and Recommendations."""
        sample = self.df.iloc[0].to_dict()
        scores = compute_composite_scores(sample)
        self.assertEqual(len(scores), 7)

        ews = compute_ews(scores)
        self.assertTrue(0 <= ews <= 100)

        band = ews_band(ews)
        self.assertIn(band, ["Safe", "Watch", "Elevated", "Critical"])

        exp = explain(scores, top_n=3)
        self.assertEqual(len(exp), 3)

        recs = recommend(exp)
        self.assertEqual(len(recs), 3)
        for r in recs:
            self.assertGreater(len(r["actions"]), 0)

    def test_07_preprocessing_pipeline(self):
        """Verifies that the raw preprocessing pipeline cleans duplicates and missing values."""
        raw_df = load_raw()
        clean_df, meta = preprocess(raw_df)
        self.assertEqual(clean_df.isna().sum().sum(), 0)
        self.assertEqual(clean_df.duplicated().sum(), 0)
        self.assertGreater(meta["rows_after"], 0)

    def test_08_dashboard_navigation_eda_explorer_removed(self):
        """Verifies EDA Explorer is removed from app.py sidebar navigation and Preprocessing & EDA is present."""
        app_path = os.path.join(BASE_DIR, "dashboard", "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()

        # EDA Explorer must NOT be an option in radio navigation
        self.assertNotIn('"🔎 EDA Explorer"', content)
        self.assertNotIn("'🔎 EDA Explorer'", content)
        # Preprocessing & EDA must be present
        self.assertIn("📊 Preprocessing & EDA", content)
        # All required pages must be present
        for page_name in [
            "🏠 Mission Control", "📊 Preprocessing & EDA", "🧠 Model Lab",
            "🔮 Predict · Explain · Recommend", "📋 Risk Ranking",
            "🧪 What-If Simulator", "💡 Business Insights"
        ]:
            self.assertIn(page_name, content)


if __name__ == "__main__":
    unittest.main()
