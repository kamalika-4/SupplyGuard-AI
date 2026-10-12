"""
SupplyGuard AI - Custom CSV Dataset Upload & Batch Prediction Test Suite
========================================================================
Tests the CSV dataset upload lifecycle:
1. Empty and corrupt file handling (graceful error handling)
2. Schema validation & compatibility assessment
3. Defensive preprocessing & feature engineering on custom datasets
4. Vectorized XGBoost disruption risk predictions and probability calculation
5. Batch MongoDB persistence to SupplyGuardDB.risk_predictions
6. Integrity check: guarantees original disk datasets remain completely unmodified
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "dashboard"))

from dashboard.risk_engine import (
    load_artifacts, load_xgb_model, validate_dataset_schema,
    batch_predict_df, REQUIRED_SHIPMENT_COLUMNS
)
from src.preprocessing import preprocess, load_raw
from src.mongo_db import (
    bulk_save_risk_predictions, get_risk_predictions,
    sanitize_for_mongo, DEFAULT_MONGO_URI, DEFAULT_DB_NAME
)


class TestCSVUploadWorkflow(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model, cls.scaler, cls.target_encoder, cls.feature_cols = load_artifacts()
        cls.xgb_model = load_xgb_model()
        cls.base_df = pd.read_csv(os.path.join(BASE_DIR, "data", "feature_engineered_data.csv"))
        cls.orig_raw_stat = os.stat(os.path.join(BASE_DIR, "data", "global_supply_chain_risk_logistics.csv")).st_mtime
        cls.orig_fe_stat = os.stat(os.path.join(BASE_DIR, "data", "feature_engineered_data.csv")).st_mtime

    def test_01_empty_and_corrupt_dataset_validation(self):
        """Verifies graceful handling of empty or None dataframes."""
        v_none = validate_dataset_schema(None)
        self.assertFalse(v_none["is_valid"])
        self.assertFalse(v_none["is_compatible"])
        self.assertIn("error", v_none)

        v_empty = validate_dataset_schema(pd.DataFrame())
        self.assertFalse(v_empty["is_valid"])
        self.assertFalse(v_empty["is_compatible"])
        self.assertIn("error", v_empty)

    def test_02_incompatible_schema_validation(self):
        """Verifies detection of non-supply-chain tabular data."""
        unrelated_df = pd.DataFrame({
            "User_ID": [1, 2, 3],
            "Username": ["alice", "bob", "charlie"],
            "Age": [25, 30, 35]
        })
        v_unrelated = validate_dataset_schema(unrelated_df)
        self.assertTrue(v_unrelated["is_valid"])
        self.assertFalse(v_unrelated["is_compatible"])
        self.assertEqual(v_unrelated["coverage_pct"], 0.0)
        self.assertEqual(len(v_unrelated["present_columns"]), 0)
        self.assertEqual(len(v_unrelated["missing_columns"]), len(REQUIRED_SHIPMENT_COLUMNS))

    def test_03_compatible_and_partial_schema_validation(self):
        """Verifies 100% match on valid supply chain data and partial compatibility detection."""
        # 1. 100% match
        sample_df = self.base_df[REQUIRED_SHIPMENT_COLUMNS].head(20).copy()
        v_full = validate_dataset_schema(sample_df)
        self.assertTrue(v_full["is_valid"])
        self.assertTrue(v_full["is_compatible"])
        self.assertEqual(v_full["coverage_pct"], 100.0)
        self.assertEqual(len(v_full["missing_columns"]), 0)

        # 2. Partial match (drop 4 non-critical columns)
        dropped_cols = ["Cost_Variance_Pct", "Single_Sourcing_Flag", "Contract_Flexibility_Score", "Fuel_Price_Index"]
        partial_df = sample_df.drop(columns=dropped_cols)
        v_part = validate_dataset_schema(partial_df)
        self.assertTrue(v_part["is_valid"])
        self.assertTrue(v_part["is_compatible"])
        self.assertGreaterEqual(v_part["coverage_pct"], 50.0)
        self.assertEqual(len(v_part["missing_columns"]), len(dropped_cols))

    def test_04_custom_dataset_preprocessing_and_cleaning(self):
        """Verifies preprocessing cleans duplicates and nulls on custom data without modifying originals."""
        dirty_df = self.base_df.head(50).copy()
        # Inject missing values first
        dirty_df.loc[0, "Lead_Time_Days"] = np.nan
        dirty_df.loc[1, "Supplier_Reliability_Score"] = np.nan
        # Inject exact duplicates
        dirty_df = pd.concat([dirty_df, dirty_df.head(5)], ignore_index=True)

        clean_df, meta = preprocess(dirty_df)
        self.assertEqual(meta["rows_before"], 55)
        self.assertEqual(meta["rows_after"], 50)
        self.assertEqual(meta["duplicates_removed"], 5)
        self.assertEqual(clean_df.isna().sum().sum(), 0)

    def test_05_batch_xgboost_predictions(self):
        """Verifies batch XGBoost predictions produce valid classes, probabilities, and top risk factors."""
        test_samples = self.base_df.head(15).copy()
        scored_df, summary = batch_predict_df(
            test_samples,
            self.xgb_model,
            self.scaler,
            self.target_encoder,
            self.feature_cols,
            base_df=self.base_df
        )

        self.assertEqual(len(scored_df), 15)
        self.assertIn("Predicted_Risk_Level", scored_df.columns)
        self.assertIn("Disruption_Probability_High", scored_df.columns)
        self.assertIn("Early_Warning_Score", scored_df.columns)
        self.assertIn("EWS_Band", scored_df.columns)
        self.assertIn("Top_Risk_Factor", scored_df.columns)

        for _, row in scored_df.iterrows():
            self.assertIn(row["Predicted_Risk_Level"], ["Low", "Medium", "High"])
            self.assertTrue(0.0 <= row["Disruption_Probability_High"] <= 1.0)
            self.assertTrue(0.0 <= row["Early_Warning_Score"] <= 100.0)
            self.assertIn(row["EWS_Band"], ["Safe", "Watch", "Elevated", "Critical"])

        self.assertEqual(summary["total"], 15)
        self.assertEqual(summary["low"] + summary["medium"] + summary["high"], 15)

    def test_06_batch_mongodb_persistence(self):
        """Verifies batch saving predictions to MongoDB risk_predictions collection."""
        now_ts = int(pd.Timestamp.now().timestamp())
        mock_preds = [
            {
                "shipment_id": f"TEST-UPLOAD-PRED-{now_ts}-{i}",
                "model_name": "XGBoostClassifier",
                "xgboost_prediction": "High" if i == 0 else "Low",
                "predicted_risk_level": "High" if i == 0 else "Low",
                "risk_probability": 0.85 if i == 0 else 0.15,
                "prediction_probabilities": {"Low": 0.1, "Medium": 0.05, "High": 0.85} if i == 0 else {"Low": 0.8, "Medium": 0.15, "High": 0.05},
                "early_warning_score": 78.5 if i == 0 else 22.0,
                "ews_band": "Critical" if i == 0 else "Safe",
                "top_risk_drivers": [{"factor": "Logistics Friction"}],
                "actual_input_values": {"Region": "East Asia", "Lead_Time_Days": 25},
                "submission_hash": f"upload_test_hash_{now_ts}_{i}",
                "dataset_source": "Uploaded_CSV_Test",
            }
            for i in range(3)
        ]

        save_res = bulk_save_risk_predictions(mock_preds)
        self.assertTrue(save_res.get("success"), f"Save failed: {save_res.get('error')}")
        self.assertEqual(save_res.get("count"), 3)

        # Retrieve one to verify presence in DB
        fetched = get_risk_predictions(filter_dict={"submission_hash": f"upload_test_hash_{now_ts}_0"})
        self.assertEqual(len(fetched), 1)
        self.assertEqual(fetched[0].get("predicted_risk_level"), "High")
        self.assertEqual(fetched[0].get("dataset_source"), "Uploaded_CSV_Test")

    def test_07_original_datasets_unchanged(self):
        """Guarantees original data files were not modified or overwritten."""
        current_raw_stat = os.stat(os.path.join(BASE_DIR, "data", "global_supply_chain_risk_logistics.csv")).st_mtime
        current_fe_stat = os.stat(os.path.join(BASE_DIR, "data", "feature_engineered_data.csv")).st_mtime
        self.assertEqual(self.orig_raw_stat, current_raw_stat)
        self.assertEqual(self.orig_fe_stat, current_fe_stat)


if __name__ == "__main__":
    unittest.main()
