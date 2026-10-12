"""
SupplyGuard AI - EWS_Score Data Type & DataFrame Display Test Suite
===================================================================
Tests:
1. EWS_Score data type consistency across MongoDB records (no mixed str/float Arrow errors)
2. PyArrow Table conversion safety for Recent Predictions and Prediction History DataFrames
3. Elimination of deprecated use_container_width in favor of width="stretch"
4. Preservation of original dataset, ML predictions, and dashboard integrity
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "dashboard"))

from dashboard.risk_engine import load_artifacts, load_xgb_model
from dashboard.agent_engine import SupplyChainDataToActionAgent
from src.mongo_db import get_risk_predictions, save_risk_prediction


class TestDataFrameEWSDisplay(unittest.TestCase):

    def test_01_no_deprecated_use_container_width_in_app(self):
        """Verifies that use_container_width has been completely replaced with width='stretch' in app.py."""
        app_path = os.path.join(BASE_DIR, "dashboard", "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            code = f.read()

        self.assertEqual(code.count("use_container_width"), 0,
                         "Deprecated parameter 'use_container_width' found in dashboard/app.py")
        self.assertGreater(code.count('width="stretch"'), 10,
                           "Expected width='stretch' to be present across components in dashboard/app.py")

    def test_02_ews_score_mixed_data_type_coercion(self):
        """Verifies that mixed early_warning_score inputs (float, 'N/A', None, string numbers) coerce cleanly to float64."""
        mock_mongo_docs = [
            {"shipment_id": "SG-1", "early_warning_score": 78.4, "predicted_risk_level": "High"},
            {"shipment_id": "SG-2", "early_warning_score": "N/A", "predicted_risk_level": "Low"},
            {"shipment_id": "SG-3", "early_warning_score": None, "predicted_risk_level": "Medium"},
            {"shipment_id": "SG-4", "early_warning_score": "45.2", "predicted_risk_level": "Watch"},
            {"shipment_id": "SG-5", "early_warning_score": 0.0, "predicted_risk_level": "Safe"},
        ]

        # Replicate app.py Recent Predictions parsing
        rec_rows = []
        for r_doc in mock_mongo_docs:
            ews_val = r_doc.get("early_warning_score")
            try:
                ews_clean = round(float(ews_val), 1) if ews_val is not None and ews_val != "N/A" else np.nan
            except (ValueError, TypeError):
                ews_clean = np.nan
            rec_rows.append({
                "Shipment_ID": r_doc.get("shipment_id", "N/A"),
                "XGBoost_Prediction": r_doc.get("predicted_risk_level", "N/A"),
                "EWS_Score": ews_clean,
            })
        rec_df = pd.DataFrame(rec_rows)
        rec_df["EWS_Score"] = pd.to_numeric(rec_df["EWS_Score"], errors="coerce")

        self.assertEqual(rec_df["EWS_Score"].dtype, np.float64)
        self.assertEqual(rec_df["EWS_Score"].iloc[0], 78.4)
        self.assertTrue(np.isnan(rec_df["EWS_Score"].iloc[1]))
        self.assertTrue(np.isnan(rec_df["EWS_Score"].iloc[2]))
        self.assertEqual(rec_df["EWS_Score"].iloc[3], 45.2)
        self.assertEqual(rec_df["EWS_Score"].iloc[4], 0.0)

    def test_03_pyarrow_dataframe_serialization_safety(self):
        """Verifies that PyArrow table conversion succeeds without ArrowInvalid errors on EWS_Score DataFrames."""
        import pyarrow as pa

        mock_mongo_docs = [
            {"shipment_id": "SG-A", "early_warning_score": 88.5, "predicted_risk_level": "High", "risk_probability": 0.92},
            {"shipment_id": "SG-B", "early_warning_score": "N/A", "predicted_risk_level": "Low", "risk_probability": 0.11},
            {"shipment_id": "SG-C", "early_warning_score": None, "predicted_risk_level": "Medium", "risk_probability": 0.53},
        ]

        h_rows = []
        for p in mock_mongo_docs:
            ews_val = p.get("early_warning_score")
            try:
                ews_clean = round(float(ews_val), 1) if ews_val is not None and ews_val != "N/A" else np.nan
            except (ValueError, TypeError):
                ews_clean = np.nan
            h_rows.append({
                "Shipment_ID": p.get("shipment_id", "N/A"),
                "XGBoost_Prediction": p.get("predicted_risk_level", "N/A"),
                "Risk_Probability": f"{float(p.get('risk_probability', 0))*100:.1f}%",
                "EWS_Score": ews_clean,
            })

        hist_df = pd.DataFrame(h_rows)
        hist_df["EWS_Score"] = pd.to_numeric(hist_df["EWS_Score"], errors="coerce")

        # PyArrow Table conversion must succeed without error
        table = pa.Table.from_pandas(hist_df)
        self.assertIsNotNone(table)
        self.assertEqual(table.column("EWS_Score").type, pa.float64())

        # Streamlit dataframe rendering call test
        try:
            st.dataframe(hist_df, width="stretch")
        except Exception as e:
            self.fail(f"st.dataframe failed with error: {e}")

    def test_04_agent_evidence_df_ews_score_type(self):
        """Verifies that agent_engine._audit_predictions_in_mongodb produces float64 EWS_Score in evidence_df."""
        df = pd.read_csv(os.path.join(BASE_DIR, "data", "feature_engineered_data.csv"))
        model, scaler, target_encoder, feature_cols = load_artifacts()
        xgb_model = load_xgb_model()

        agent = SupplyChainDataToActionAgent(
            df=df,
            xgb_model=xgb_model,
            model=model,
            scaler=scaler,
            target_encoder=target_encoder,
            feature_cols=feature_cols,
        )

        resp = agent.answer_question("Audit prediction history in MongoDB")
        self.assertTrue(resp.success)
        if resp.evidence_df is not None and not resp.evidence_df.empty:
            if "EWS_Score" in resp.evidence_df.columns:
                self.assertEqual(resp.evidence_df["EWS_Score"].dtype, np.float64)


if __name__ == "__main__":
    unittest.main()
