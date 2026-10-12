"""
SupplyGuard AI - MongoDB Integration Test Suite
================================================
Tests MongoDB connection, CRUD operations, type serialization,
real ML prediction persistence, and error handling when offline.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "dashboard"))

from src.mongo_db import (
    check_mongo_connection,
    save_shipment,
    save_risk_prediction,
    get_shipments,
    get_risk_predictions,
    bulk_save_shipments,
    sanitize_for_mongo,
    close_mongo_clients,
    DEFAULT_MONGO_URI,
    DEFAULT_DB_NAME,
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
)


class TestMongoDBIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_path = os.path.join(BASE_DIR, "data", "feature_engineered_data.csv")
        cls.df = pd.read_csv(cls.data_path)
        cls.model, cls.scaler, cls.target_encoder, cls.feature_cols = load_artifacts()
        cls.xgb_model = load_xgb_model()

    @classmethod
    def tearDownClass(cls):
        close_mongo_clients()

    def test_01_connection_check(self):
        """Verifies local MongoDB connects to SupplyGuardDB on mongodb://localhost:27017/."""
        connected, info, err = check_mongo_connection()
        self.assertTrue(connected, f"MongoDB connection failed: {err}")
        self.assertEqual(info.get("database"), DEFAULT_DB_NAME)
        self.assertEqual(info.get("uri"), DEFAULT_MONGO_URI)
        self.assertIn("shipments", info.get("collections", []))
        self.assertIn("risk_predictions", info.get("collections", []))

    def test_02_type_sanitization(self):
        """Verifies numpy and pandas types are properly converted to BSON-compatible types."""
        raw_test = {
            "int_val": np.int64(42),
            "float_val": np.float64(3.14159),
            "nan_val": np.nan,
            "arr_val": np.array([1, 2, 3]),
            "timestamp": pd.Timestamp("2026-10-12 12:00:00"),
        }
        clean = sanitize_for_mongo(raw_test)
        self.assertIsInstance(clean["int_val"], int)
        self.assertIsInstance(clean["float_val"], float)
        self.assertIsNone(clean["nan_val"])
        self.assertIsInstance(clean["arr_val"], list)
        self.assertEqual(clean["arr_val"], [1, 2, 3])

    def test_03_save_and_retrieve_shipment(self):
        """Verifies saving an actual shipment record and retrieving it from MongoDB."""
        sample_shipment = self.df.iloc[0].to_dict()
        sid = sample_shipment.get("Shipment_ID", "SH-00001")
        
        save_res = save_shipment(sample_shipment, status="In Transit")
        self.assertTrue(save_res.get("success"), f"Save shipment failed: {save_res.get('error')}")
        self.assertEqual(save_res.get("shipment_id"), sid)

        # Retrieve shipment
        retrieved = get_shipments(filter_dict={"shipment_id": sid}, limit=5)
        self.assertGreaterEqual(len(retrieved), 1)
        doc = retrieved[0]
        self.assertEqual(doc.get("shipment_id"), sid)
        self.assertIn("status", doc)
        self.assertIn("Region", doc)

    def test_04_save_and_retrieve_real_ml_prediction(self):
        """Verifies running real XGBoost prediction and persisting the result in risk_predictions."""
        self.assertIsNotNone(self.xgb_model)
        sample = self.df.iloc[1].to_dict()
        sid = sample.get("Shipment_ID", "SH-00002")

        # Real ML prediction
        label, proba = predict_ml(sample, self.xgb_model, self.scaler, self.target_encoder, self.feature_cols)
        scores = compute_composite_scores(sample)
        ews = compute_ews(scores)
        band = ews_band(ews)
        exp = explain(scores)
        recs = recommend(exp)

        pred_payload = {
            "shipment_id": sid,
            "model_name": "XGBoostClassifier",
            "predicted_risk_level": label,
            "prediction_probabilities": proba,
            "early_warning_score": round(ews, 2),
            "ews_band": band,
            "top_risk_drivers": exp,
            "recommendations": [r["factor"] for r in recs],
            "prediction_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        save_res = save_risk_prediction(pred_payload)
        self.assertTrue(save_res.get("success"), f"Save prediction failed: {save_res.get('error')}")
        self.assertIn("prediction_id", save_res)

        # Retrieve prediction
        retrieved = get_risk_predictions(filter_dict={"shipment_id": sid}, limit=5)
        self.assertGreaterEqual(len(retrieved), 1)
        pdoc = retrieved[0]
        self.assertEqual(pdoc.get("shipment_id"), sid)
        self.assertEqual(pdoc.get("predicted_risk_level"), label)
        self.assertEqual(pdoc.get("model_name"), "XGBoostClassifier")
        self.assertIn("prediction_probabilities", pdoc)
        self.assertAlmostEqual(sum(pdoc["prediction_probabilities"].values()), 1.0, places=2)

    def test_05_bulk_save_shipments(self):
        """Verifies bulk saving multiple actual shipment records from the dataset."""
        records = self.df.iloc[2:10].to_dict(orient="records")
        res = bulk_save_shipments(records)
        self.assertTrue(res.get("success"), f"Bulk save failed: {res.get('error')}")
        self.assertEqual(res.get("total_processed"), 8)

    def test_06_graceful_offline_handling(self):
        """Verifies graceful handling without crashing when MongoDB is unreachable."""
        offline_uri = "mongodb://localhost:27019/"
        conn, info, err = check_mongo_connection(uri=offline_uri, timeout_ms=300)
        self.assertFalse(conn)
        self.assertIsNotNone(err)

        save_ship_res = save_shipment({"shipment_id": "TEST-OFFLINE"}, uri=offline_uri)
        self.assertFalse(save_ship_res["success"])
        self.assertIn("error", save_ship_res)

        save_pred_res = save_risk_prediction({"shipment_id": "TEST-OFFLINE"}, uri=offline_uri)
        self.assertFalse(save_pred_res["success"])

        shipments = get_shipments(uri=offline_uri)
        self.assertEqual(shipments, [])

        predictions = get_risk_predictions(uri=offline_uri)
        self.assertEqual(predictions, [])

    def test_07_prediction_deduplication_and_shap_summary(self):
        """Verifies duplicate submission prevention and persistence of SHAP summary."""
        test_payload = {
            "shipment_id": "SH-TEST-DEDUP-001",
            "model_name": "XGBoostClassifier",
            "predicted_risk_level": "High",
            "risk_probability": 0.88,
            "prediction_probabilities": {"Low": 0.02, "Medium": 0.10, "High": 0.88},
            "early_warning_score": 84.5,
            "ews_band": "Critical",
            "shap_summary": {
                "base_value": -0.85,
                "margin_value": 1.25,
                "sum_shap": 2.10,
                "disruption_prob": 0.88,
                "top_risk_drivers": [{"factor": "Geo Weather", "shap_value": 0.95}],
            },
            "shipment_inputs": {"Country": "Vietnam", "Lead_Time_Days": 25},
            "submission_hash": "test_dedup_hash_12345",
        }

        # 1. First insert
        res1 = save_risk_prediction(test_payload, prevent_duplicates=True)
        self.assertTrue(res1.get("success"))
        self.assertIn("prediction_id", res1)

        # 2. Duplicate submission with same submission_hash
        res2 = save_risk_prediction(test_payload, prevent_duplicates=True)
        self.assertTrue(res2.get("success"))
        self.assertTrue(res2.get("duplicate"))
        self.assertEqual(res1.get("prediction_id"), res2.get("prediction_id"))

        # 3. Retrieve and verify SHAP summary & inputs
        retrieved = get_risk_predictions(filter_dict={"submission_hash": "test_dedup_hash_12345"}, limit=5)
        self.assertEqual(len(retrieved), 1)
        doc = retrieved[0]
        self.assertIn("shap_summary", doc)
        self.assertIn("shipment_inputs", doc)
        self.assertEqual(doc["shap_summary"]["base_value"], -0.85)
        self.assertEqual(doc["shipment_inputs"]["Country"], "Vietnam")


if __name__ == "__main__":
    unittest.main()
