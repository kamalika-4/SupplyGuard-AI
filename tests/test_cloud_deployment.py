"""
SupplyGuard AI - Streamlit Community Cloud Deployment Test Suite
=================================================================
Tests cloud environment detection, graceful MongoDB bypass,
root entry-point delegator, requirements.txt, and page stability.
"""

import os
import sys
import unittest
from unittest.mock import patch
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

from src.mongo_db import (
    is_cloud_environment,
    is_mongo_persistence_disabled,
    check_mongo_connection,
    save_risk_prediction,
    bulk_save_risk_predictions,
    save_shipment,
    bulk_save_shipments,
    get_shipments,
    get_risk_predictions,
    close_mongo_clients,
)
from dashboard.risk_engine import (
    load_artifacts,
    load_xgb_model,
    predict_ml,
    compute_composite_scores,
    compute_ews,
    ews_band,
    explain,
    recommend,
    batch_predict_df,
)
from dashboard.shap_engine import (
    compute_local_shap,
    plot_shap_waterfall,
    compute_global_shap_data,
    plot_shap_summary_bar,
)
from dashboard.agent_engine import SupplyChainDataToActionAgent


class TestCloudDeployment(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_path = os.path.join(BASE_DIR, "data", "feature_engineered_data.csv")
        cls.df = pd.read_csv(cls.data_path)
        cls.model, cls.scaler, cls.target_encoder, cls.feature_cols, cls.xgb_model = load_artifacts(include_xgb=True)
        cls.agent = SupplyChainDataToActionAgent(cls.df, cls.xgb_model, cls.model, cls.scaler, cls.target_encoder, cls.feature_cols)

    @classmethod
    def tearDownClass(cls):
        close_mongo_clients()

    def test_01_root_entry_point_exists(self):
        """Verifies root app.py exists and references dashboard/app.py for Streamlit Cloud."""
        root_app = os.path.join(BASE_DIR, "app.py")
        self.assertTrue(os.path.exists(root_app), "Root app.py must exist for Streamlit Cloud.")
        with open(root_app, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("dashboard", content)
        self.assertIn("app.py", content)

    def test_02_requirements_txt_completeness(self):
        """Verifies requirements.txt specifies all essential dependencies including pyarrow."""
        req_path = os.path.join(BASE_DIR, "requirements.txt")
        self.assertTrue(os.path.exists(req_path))
        with open(req_path, "r", encoding="utf-8") as f:
            reqs = [line.strip().split(">=")[0].split("<")[0].lower() for line in f if line.strip() and not line.startswith("#")]
        expected = ["pandas", "numpy", "scikit-learn", "xgboost", "streamlit", "plotly", "joblib", "shap", "pymongo", "pyarrow"]
        for exp in expected:
            self.assertIn(exp, reqs, f"Missing {exp} in requirements.txt")

    def test_03_local_environment_mode(self):
        """Verifies that locally, persistence is enabled for localhost:27017."""
        # In current test runner without cloud env vars set:
        if not is_cloud_environment():
            self.assertFalse(is_mongo_persistence_disabled("mongodb://localhost:27017/"))

    def test_04_cloud_environment_detection_and_graceful_bypass(self):
        """Verifies that under Streamlit Cloud, MongoDB persistence is gracefully bypassed without connection errors."""
        with patch.dict(os.environ, {"IS_STREAMLIT_CLOUD": "true"}):
            self.assertTrue(is_cloud_environment())
            self.assertTrue(is_mongo_persistence_disabled("mongodb://localhost:27017/"))

            # 1. Connection check returns False, info with cloud_mode, and NO error
            connected, info, err = check_mongo_connection("mongodb://localhost:27017/")
            self.assertFalse(connected)
            self.assertIsNone(err)
            self.assertTrue(info.get("cloud_mode"))
            self.assertEqual(info.get("status"), "Disabled (Streamlit Cloud Mode)")

            # 2. Saving single prediction returns disabled without socket failure or false success
            pred_res = save_risk_prediction({"shipment_id": "TEST-CLOUD-01", "predicted_risk_level": "High"})
            self.assertFalse(pred_res.get("success"))
            self.assertTrue(pred_res.get("cloud_mode"))
            self.assertTrue(pred_res.get("disabled"))
            self.assertNotIn("inserted_id", pred_res)
            self.assertIn("Streamlit Cloud Mode", pred_res.get("message", ""))

            # 3. Bulk saving returns disabled without socket failure
            b_res = bulk_save_risk_predictions([{"shipment_id": "TEST-CLOUD-01"}])
            self.assertFalse(b_res.get("success"))
            self.assertTrue(b_res.get("cloud_mode"))
            self.assertTrue(b_res.get("disabled"))

            # 4. Save shipment returns disabled
            s_res = save_shipment({"shipment_id": "TEST-CLOUD-01"})
            self.assertFalse(s_res.get("success"))
            self.assertTrue(s_res.get("cloud_mode"))

            # 5. Queries return empty list immediately without hanging
            self.assertEqual(get_shipments(), [])
            self.assertEqual(get_risk_predictions(), [])

    def test_05_remote_mongodb_in_cloud_allowed(self):
        """Verifies that if a remote MongoDB Atlas URI is configured in cloud, it is not flagged as disabled."""
        with patch.dict(os.environ, {"IS_STREAMLIT_CLOUD": "true"}):
            atlas_uri = "mongodb+srv://admin:pass@cluster0.mongodb.net/SupplyGuardDB"
            self.assertFalse(is_mongo_persistence_disabled(atlas_uri))

    def test_06_xgboost_and_shap_in_cloud(self):
        """Verifies XGBoost prediction and local/global SHAP computation run accurately without exceptions."""
        sample = self.df.iloc[0].to_dict()
        scores = compute_composite_scores(sample)
        ews = compute_ews(scores)
        band = ews_band(ews)
        self.assertIn(band, ["Safe", "Watch", "Elevated", "Critical"])

        label, proba = predict_ml(sample, self.xgb_model, self.scaler, self.target_encoder, self.feature_cols)
        self.assertIn(label, ["Low", "Medium", "High"])
        self.assertIn("High", proba)

        local_shap = compute_local_shap(self.xgb_model, sample, self.feature_cols, target_encoder=self.target_encoder)
        self.assertIn("base_value", local_shap)
        self.assertIn("sum_shap", local_shap)
        self.assertIn("margin_value", local_shap)

        # Plotly waterfall generation
        fig = plot_shap_waterfall(local_shap)
        self.assertIsNotNone(fig)

        # Global SHAP
        g_shap = compute_global_shap_data(self.xgb_model, self.df.head(50), self.feature_cols, target_encoder=self.target_encoder)
        self.assertIn("ranking", g_shap)
        bar_fig = plot_shap_summary_bar(g_shap)
        self.assertIsNotNone(bar_fig)

    def test_07_batch_predictions_resilience(self):
        """Verifies batch XGBoost predictions handle normal, empty, and partial datasets safely."""
        # 1. Normal batch
        scored_df, summary = batch_predict_df(self.df.head(20), self.xgb_model, self.scaler, self.target_encoder, self.feature_cols, base_df=self.df)
        self.assertEqual(len(scored_df), 20)
        self.assertEqual(summary["total"], 20)

        # 2. Empty dataset
        empty_df = pd.DataFrame()
        empty_scored, empty_sum = batch_predict_df(empty_df, self.xgb_model, self.scaler, self.target_encoder, self.feature_cols, base_df=self.df)
        self.assertEqual(len(empty_scored), 0)
        self.assertEqual(empty_sum["total"], 0)

    def test_08_agent_cloud_fallback(self):
        """Verifies AI Agent falls back to active dataset when MongoDB is offline/disabled in cloud."""
        with patch.dict(os.environ, {"IS_STREAMLIT_CLOUD": "true"}):
            sid = self.df.iloc[0]["Shipment_ID"]
            diag = self.agent.analyze_shipment(sid)
            self.assertTrue(diag.success)
            self.assertIn(sid, diag.summary)
            self.assertFalse(diag.evidence_df.empty)

            # Audit prediction history in cloud mode
            audit_res = self.agent.audit_prediction_history()
            self.assertIn("Cloud Deployment Mode", audit_res.summary)


if __name__ == "__main__":
    unittest.main()
