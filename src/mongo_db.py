"""
SupplyGuard AI - MongoDB Database Connector & Integration
==========================================================
Connects to local MongoDB instance (default: mongodb://localhost:27017/ -> SupplyGuardDB)
Manages collections:
  - shipments: actual shipment records and operational logistics data
  - risk_predictions: real ML model prediction results, probabilities, and risk telemetry

Handles errors gracefully with low timeout to prevent dashboard freezing when offline.
"""

import os
import datetime
import hashlib
import json
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    import pymongo
    from pymongo.errors import PyMongoError, ConnectionFailure, ServerSelectionTimeoutError
    PYMONGO_AVAILABLE = True
except ImportError:
    pymongo = None
    PYMONGO_AVAILABLE = False
    PyMongoError = Exception
    ConnectionFailure = Exception
    ServerSelectionTimeoutError = Exception

DEFAULT_MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DEFAULT_DB_NAME = os.getenv("MONGODB_DATABASE", "SupplyGuardDB")
SHIPMENTS_COLLECTION = "shipments"
PREDICTIONS_COLLECTION = "risk_predictions"


def sanitize_for_mongo(val: Any) -> Any:
    """Recursively converts numpy, pandas, and other non-BSON types to standard Python primitives."""
    if val is None:
        return None
    if isinstance(val, (bool, str, bytes, datetime.datetime, datetime.date)):
        return val
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    if isinstance(val, (np.ndarray, list, tuple, set)):
        return [sanitize_for_mongo(x) for x in val]
    if isinstance(val, dict):
        return {str(k): sanitize_for_mongo(v) for k, v in val.items()}
    if hasattr(val, "to_dict"):
        return sanitize_for_mongo(val.to_dict())
    if pd.isna(val):
        return None
    return str(val)


_CLIENTS: Dict[Tuple[str, int], Any] = {}


def get_mongo_client(uri: Optional[str] = None, timeout_ms: int = 2000):
    """Returns a cached MongoClient instance with low serverSelectionTimeoutMS for responsive fallback."""
    if not PYMONGO_AVAILABLE:
        return None
    uri = uri or DEFAULT_MONGO_URI
    cache_key = (uri, timeout_ms)
    client = _CLIENTS.get(cache_key)
    if client is not None:
        return client

    try:
        new_client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=timeout_ms, connectTimeoutMS=timeout_ms)
        _CLIENTS[cache_key] = new_client
        return new_client
    except Exception:
        return None


def close_mongo_clients():
    """Safely closes all cached MongoClients."""
    for client in list(_CLIENTS.values()):
        try:
            client.close()
        except Exception:
            pass
    _CLIENTS.clear()


def check_mongo_connection(
    uri: Optional[str] = None,
    db_name: Optional[str] = None,
    timeout_ms: int = 2000,
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Checks MongoDB connectivity and returns (is_connected, info_dict, error_message).
    Never raises an uncaught exception.
    """
    if not PYMONGO_AVAILABLE:
        return False, {}, "pymongo package is not installed."

    uri = uri or DEFAULT_MONGO_URI
    db_name = db_name or DEFAULT_DB_NAME

    try:
        client = get_mongo_client(uri, timeout_ms=timeout_ms)
        if client is None:
            return False, {}, "Failed to create MongoClient."
        # Ping the server
        client.admin.command("ping")
        server_info = client.server_info()
        db = client[db_name]
        collections = db.list_collection_names()
        
        # Count documents safely
        shipments_count = db[SHIPMENTS_COLLECTION].count_documents({}) if SHIPMENTS_COLLECTION in collections else 0
        predictions_count = db[PREDICTIONS_COLLECTION].count_documents({}) if PREDICTIONS_COLLECTION in collections else 0

        info = {
            "uri": uri,
            "database": db_name,
            "version": server_info.get("version", "unknown"),
            "collections": collections,
            "shipments_count": shipments_count,
            "predictions_count": predictions_count,
            "status": "Connected",
        }
        return True, info, None
    except Exception as e:
        cache_key = (uri, timeout_ms)
        failed_client = _CLIENTS.pop(cache_key, None)
        if failed_client:
            try:
                failed_client.close()
            except Exception:
                pass
        return False, {"uri": uri, "database": db_name, "status": "Offline"}, str(e)


def get_database_with_error(
    uri: Optional[str] = None,
    db_name: Optional[str] = None,
    timeout_ms: int = 2000,
) -> Tuple[Optional[Any], Optional[str]]:
    """Returns the database handle and error message if unavailable."""
    if not PYMONGO_AVAILABLE:
        return None, "pymongo package is not installed."
    uri = uri or DEFAULT_MONGO_URI
    db_name = db_name or DEFAULT_DB_NAME
    try:
        client = get_mongo_client(uri, timeout_ms=timeout_ms)
        if client is None:
            return None, f"Failed to initialize MongoClient for '{uri}'."
        client.admin.command("ping")
        return client[db_name], None
    except Exception as e:
        cache_key = (uri, timeout_ms)
        failed_client = _CLIENTS.pop(cache_key, None)
        if failed_client:
            try:
                failed_client.close()
            except Exception:
                pass
        return None, str(e)


def get_database(uri: Optional[str] = None, db_name: Optional[str] = None, timeout_ms: int = 2000):
    """Returns the database handle if connected, or None if unavailable."""
    db, _ = get_database_with_error(uri, db_name, timeout_ms=timeout_ms)
    return db


def save_shipment(
    shipment_data: Dict[str, Any],
    uri: Optional[str] = None,
    db_name: Optional[str] = None,
    status: Optional[str] = None,
    timeout_ms: int = 2000,
) -> Dict[str, Any]:
    """
    Saves or updates an actual shipment record in the 'shipments' collection.
    Upserts by shipment_id / Shipment_ID if present.
    """
    db, conn_err = get_database_with_error(uri, db_name, timeout_ms=timeout_ms)
    if db is None:
        return {"success": False, "error": f"MongoDB server is not reachable: {conn_err or 'Connection failed'}"}

    try:
        clean = sanitize_for_mongo(shipment_data)
        sid = clean.get("shipment_id") or clean.get("Shipment_ID")
        if not sid:
            sid = f"SH-{int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)}"
            clean["shipment_id"] = sid

        # Normalize common fields
        clean["shipment_id"] = sid
        clean["Shipment_ID"] = sid
        if status:
            clean["status"] = status
        elif "status" not in clean:
            clean["status"] = "In Transit"

        clean["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if "created_at" not in clean:
            clean["created_at"] = clean["updated_at"]

        # Upsert by shipment_id so duplicates are updated gracefully
        res = db[SHIPMENTS_COLLECTION].update_one(
            {"$or": [{"shipment_id": sid}, {"Shipment_ID": sid}]},
            {"$set": clean},
            upsert=True,
        )

        doc_id = str(res.upserted_id) if res.upserted_id else sid
        return {
            "success": True,
            "shipment_id": sid,
            "doc_id": doc_id,
            "matched_count": res.matched_count,
            "modified_count": res.modified_count,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def save_risk_prediction(
    prediction_data: Dict[str, Any],
    uri: Optional[str] = None,
    db_name: Optional[str] = None,
    prevent_duplicates: bool = True,
    timeout_ms: int = 2000,
) -> Dict[str, Any]:
    """
    Saves a real ML prediction result into the 'risk_predictions' collection.
    Never stores fake data; expects computed ML label, probabilities, and telemetry.
    Supports duplicate prevention via submission_hash matching if prevent_duplicates=True.
    Verifies the connection and actual insert_one() execution.
    """
    uri = uri or DEFAULT_MONGO_URI
    db_name = db_name or DEFAULT_DB_NAME

    db, conn_err = get_database_with_error(uri, db_name, timeout_ms=timeout_ms)
    if db is None:
        return {
            "success": False,
            "error": f"MongoDB connection to {uri} ({db_name}) failed: {conn_err or 'Server is unreachable.'}",
        }

    try:
        clean = dict(sanitize_for_mongo(prediction_data))
        clean.pop("_id", None)  # Ensure a new document ID is generated

        sid = clean.get("shipment_id") or clean.get("Shipment_ID", "UNKNOWN")
        clean["shipment_id"] = sid
        clean["Shipment_ID"] = sid

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if "created_at" not in clean:
            clean["created_at"] = now_iso
        clean["timestamp"] = clean.get("timestamp") or clean.get("prediction_timestamp") or clean["created_at"]
        clean["prediction_timestamp"] = clean.get("prediction_timestamp") or clean["timestamp"]

        # Ensure consistent prediction and input field keys
        pred_label = clean.get("predicted_risk_level") or clean.get("xgboost_prediction")
        if pred_label:
            clean["predicted_risk_level"] = pred_label
            clean["xgboost_prediction"] = pred_label

        inputs = clean.get("actual_input_values") or clean.get("shipment_inputs") or clean.get("input_values") or {}
        if inputs:
            clean["actual_input_values"] = inputs
            clean["shipment_inputs"] = inputs
            clean["input_values"] = inputs

        shap_exp = clean.get("shap_explanation") or clean.get("shap_summary")
        if shap_exp:
            clean["shap_explanation"] = shap_exp
            clean["shap_summary"] = shap_exp

        # Ensure submission_hash for deduplication
        if "submission_hash" not in clean:
            h_str = f"{sid}_{clean.get('predicted_risk_level')}_{json.dumps(inputs, sort_keys=True, default=str)}"
            clean["submission_hash"] = hashlib.sha256(h_str.encode()).hexdigest()[:16]

        # Check for duplicates if enabled
        if prevent_duplicates and clean.get("submission_hash"):
            existing = db[PREDICTIONS_COLLECTION].find_one({"submission_hash": clean["submission_hash"]})
            if existing:
                return {
                    "success": True,
                    "duplicate": True,
                    "shipment_id": sid,
                    "prediction_id": str(existing["_id"]),
                    "inserted_id": str(existing["_id"]),
                    "message": "Duplicate submission prevented. Existing prediction record preserved.",
                }

        # Actual insert_one() execution
        res = db[PREDICTIONS_COLLECTION].insert_one(clean)
        if not getattr(res, "acknowledged", False) or getattr(res, "inserted_id", None) is None:
            return {
                "success": False,
                "error": "MongoDB insert_one() execution was not acknowledged by the server.",
            }

        inserted_id_str = str(res.inserted_id)
        return {
            "success": True,
            "duplicate": False,
            "shipment_id": sid,
            "prediction_id": inserted_id_str,
            "inserted_id": inserted_id_str,
            "database": db_name,
            "collection": PREDICTIONS_COLLECTION,
            "acknowledged": True,
        }
    except Exception as e:
        return {"success": False, "error": f"MongoDB insert_one() execution failed: {e}"}


def get_shipments(
    filter_dict: Optional[Dict[str, Any]] = None,
    limit: int = 100,
    uri: Optional[str] = None,
    db_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieves saved shipment records from MongoDB."""
    db = get_database(uri, db_name)
    if db is None:
        return []

    try:
        query = filter_dict or {}
        cursor = db[SHIPMENTS_COLLECTION].find(query).sort("_id", -1).limit(limit)
        results = []
        for doc in cursor:
            doc["_id"] = str(doc["_id"])
            results.append(doc)
        return results
    except Exception:
        return []


def get_risk_predictions(
    filter_dict: Optional[Dict[str, Any]] = None,
    limit: int = 100,
    uri: Optional[str] = None,
    db_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieves saved ML prediction records from MongoDB."""
    db = get_database(uri, db_name)
    if db is None:
        return []

    try:
        query = filter_dict or {}
        cursor = db[PREDICTIONS_COLLECTION].find(query).sort([("_id", -1)]).limit(limit)
        results = []
        for doc in cursor:
            doc["_id"] = str(doc["_id"])
            results.append(doc)
        return results
    except Exception:
        return []


def bulk_save_shipments(
    records: List[Dict[str, Any]],
    uri: Optional[str] = None,
    db_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Bulk imports or updates shipment records into MongoDB."""
    db = get_database(uri, db_name)
    if db is None:
        return {"success": False, "error": "MongoDB server is not reachable."}

    if not records:
        return {"success": True, "count": 0}

    try:
        ops = []
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        for r in records:
            clean = sanitize_for_mongo(r)
            sid = clean.get("shipment_id") or clean.get("Shipment_ID")
            if not sid:
                continue
            clean["shipment_id"] = sid
            clean["Shipment_ID"] = sid
            if "status" not in clean:
                clean["status"] = "In Transit"
            clean["updated_at"] = now
            ops.append(
                pymongo.UpdateOne(
                    {"$or": [{"shipment_id": sid}, {"Shipment_ID": sid}]},
                    {"$set": clean},
                    upsert=True,
                )
            )

        if ops:
            result = db[SHIPMENTS_COLLECTION].bulk_write(ops, ordered=False)
            return {
                "success": True,
                "upserted": len(result.upserted_ids),
                "modified": result.modified_count,
                "matched": result.matched_count,
                "total_processed": len(ops),
            }
        return {"success": True, "count": 0}
    except Exception as e:
        return {"success": False, "error": str(e)}


def bulk_save_risk_predictions(
    predictions: List[Dict[str, Any]],
    uri: Optional[str] = None,
    db_name: Optional[str] = None,
    timeout_ms: int = 3000,
) -> Dict[str, Any]:
    """
    Bulk saves ML risk prediction records into the 'risk_predictions' collection.
    Verifies database connection and insert_many() execution.
    """
    uri = uri or DEFAULT_MONGO_URI
    db_name = db_name or DEFAULT_DB_NAME

    db, conn_err = get_database_with_error(uri, db_name, timeout_ms=timeout_ms)
    if db is None:
        return {
            "success": False,
            "error": f"MongoDB server is not reachable: {conn_err or 'Connection failed'}",
        }

    if not predictions:
        return {"success": True, "count": 0, "inserted_ids": []}

    try:
        clean_docs = []
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        for p in predictions:
            clean = dict(sanitize_for_mongo(p))
            clean.pop("_id", None)
            sid = clean.get("shipment_id") or clean.get("Shipment_ID", "UNKNOWN")
            clean["shipment_id"] = sid
            clean["Shipment_ID"] = sid
            if "timestamp" not in clean and "created_at" not in clean:
                clean["timestamp"] = now_iso
                clean["created_at"] = now_iso
            clean_docs.append(clean)

        res = db[PREDICTIONS_COLLECTION].insert_many(clean_docs, ordered=False)
        return {
            "success": True,
            "count": len(res.inserted_ids),
            "inserted_ids": [str(i) for i in res.inserted_ids],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

