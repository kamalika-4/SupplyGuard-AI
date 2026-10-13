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

__all__ = [
    "DEFAULT_MONGO_URI",
    "DEFAULT_DB_NAME",
    "SHIPMENTS_COLLECTION",
    "PREDICTIONS_COLLECTION",
    "PYMONGO_AVAILABLE",
    "get_configured_mongo_uri",
    "get_configured_db_name",
    "is_cloud_environment",
    "is_mongo_persistence_disabled",
    "sanitize_for_mongo",
    "get_candidate_uris",
    "get_mongo_client",
    "close_mongo_clients",
    "check_mongo_connection",
    "get_database_with_error",
    "get_database",
    "save_shipment",
    "save_risk_prediction",
    "get_shipments",
    "get_risk_predictions",
    "bulk_save_shipments",
    "bulk_save_risk_predictions",
]


def get_configured_mongo_uri(uri: Optional[str] = None) -> str:
    """
    Resolves the MongoDB connection URI in order of priority:
    1. An explicitly provided remote or non-default URI (e.g. mongodb+srv:// or custom port).
    2. Streamlit Cloud Secrets: st.secrets["MONGODB_URI"] (if configured).
    3. Environment variable: os.getenv("MONGODB_URI").
    4. Explicit uri if provided (e.g. local loopback).
    5. Default local MongoDB URI: DEFAULT_MONGO_URI (mongodb://localhost:27017/).
    """
    # 1. If caller explicitly passed a non-loopback URI, respect it
    if uri and ("localhost" not in uri) and ("127.0.0.1" not in uri):
        return uri.strip()

    # 2. Check Streamlit secrets first (for Streamlit Community Cloud or local .streamlit/secrets.toml)
    try:
        import streamlit as _st
        if hasattr(_st, "secrets") and "MONGODB_URI" in _st.secrets:
            secret_uri = _st.secrets["MONGODB_URI"]
            if secret_uri and str(secret_uri).strip():
                return str(secret_uri).strip()
    except Exception:
        pass

    # 3. Check environment variable
    env_uri = os.getenv("MONGODB_URI")
    if env_uri and env_uri.strip():
        return env_uri.strip()

    # 4. If caller provided an explicit URI (even loopback), return it
    if uri and uri.strip():
        return uri.strip()

    return DEFAULT_MONGO_URI


def get_configured_db_name(db_name: Optional[str] = None) -> str:
    """Resolves the database name from argument, Streamlit secrets, environment, or default."""
    if db_name and db_name.strip():
        return db_name.strip()
    try:
        import streamlit as _st
        if hasattr(_st, "secrets") and "MONGODB_DATABASE" in _st.secrets:
            secret_db = _st.secrets["MONGODB_DATABASE"]
            if secret_db and str(secret_db).strip():
                return str(secret_db).strip()
    except Exception:
        pass
    env_db = os.getenv("MONGODB_DATABASE")
    if env_db and env_db.strip():
        return env_db.strip()
    return DEFAULT_DB_NAME


def is_cloud_environment() -> bool:
    """
    Detects if running on Streamlit Community Cloud or remote container environment.
    Checks environment variables, container paths, and Streamlit secrets.
    """
    for var in (
        "STREAMLIT_SHARING_MODE",
        "IS_STREAMLIT_CLOUD",
        "STREAMLIT_COMMUNITY_CLOUD",
        "STREAMLIT_CLOUD",
        "SPACE_ID",
        "RENDER",
        "FLY_APP_NAME",
        "RAILWAY_STATIC_URL",
    ):
        if os.getenv(var):
            return True

    # Common Streamlit Community Cloud container directory markers
    if os.path.exists("/mount/src") or os.path.exists("/home/appuser"):
        return True

    # Streamlit secrets check
    try:
        import streamlit as _st
        if hasattr(_st, "secrets"):
            if _st.secrets.get("IS_STREAMLIT_CLOUD", False):
                return True
            if _st.secrets.get("STREAMLIT_CLOUD", False):
                return True
    except Exception:
        pass

    return False


def is_mongo_persistence_disabled(uri: Optional[str] = None) -> bool:
    """
    Determines if MongoDB persistence should be gracefully disabled.
    Only disabled in cloud deployment when the resolved URI is localhost/127.0.0.1
    and NO remote MongoDB URI (such as Atlas in st.secrets['MONGODB_URI']) is configured.
    When running locally or when a remote MongoDB Atlas URI is configured, returns False.
    """
    target_uri = get_configured_mongo_uri(uri)

    # If it is a remote cluster URI (e.g. Atlas mongodb+srv://), persistence is NOT disabled
    is_loopback = ("localhost" in target_uri) or ("127.0.0.1" in target_uri)
    if is_cloud_environment() and is_loopback:
        return True
    return False



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


def get_candidate_uris(uri: str) -> List[str]:
    """Generates candidate URIs to gracefully bridge IPv4/IPv6 loopback bindings on Windows."""
    candidates = [uri]
    if "localhost" in uri:
        candidates.append(uri.replace("localhost", "127.0.0.1"))
    elif "127.0.0.1" in uri:
        candidates.append(uri.replace("127.0.0.1", "localhost"))
    return candidates


def get_mongo_client(uri: Optional[str] = None, timeout_ms: int = 4000):
    """Returns a cached MongoClient instance with serverSelectionTimeoutMS for responsive fallback."""
    if not PYMONGO_AVAILABLE:
        return None
    uri = get_configured_mongo_uri(uri)
    cache_key = (uri, timeout_ms)
    client = _CLIENTS.get(cache_key)
    if client is not None:
        try:
            client.admin.command("ping")
            return client
        except Exception:
            _CLIENTS.pop(cache_key, None)
            try:
                client.close()
            except Exception:
                pass

    candidates = get_candidate_uris(uri)
    for cand in candidates:
        try:
            kwargs = {
                "serverSelectionTimeoutMS": timeout_ms,
                "connectTimeoutMS": timeout_ms,
            }
            if ("mongodb+srv://" in cand or "ssl=true" in cand.lower() or "tls=true" in cand.lower()):
                try:
                    import certifi
                    kwargs["tlsCAFile"] = certifi.where()
                except Exception:
                    pass
            new_client = pymongo.MongoClient(cand, **kwargs)
            new_client.admin.command("ping")
            _CLIENTS[cache_key] = new_client
            return new_client
        except Exception:
            try:
                new_client.close()
            except Exception:
                pass
            continue

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
    timeout_ms: int = 4000,
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Checks MongoDB connectivity and returns (is_connected, info_dict, error_message).
    Never raises an uncaught exception.
    """
    if not PYMONGO_AVAILABLE:
        return False, {}, "pymongo package is not installed."

    resolved_uri = get_configured_mongo_uri(uri)
    resolved_db = get_configured_db_name(db_name)

    if is_mongo_persistence_disabled(uri):
        info = {
            "uri": "mongodb://localhost:27017/ (disabled in cloud)",
            "database": resolved_db,
            "status": "Disabled (Streamlit Cloud Mode)",
            "cloud_mode": True,
            "collections": [],
            "shipments_count": 0,
            "predictions_count": 0,
            "message": "MongoDB persistence is gracefully disabled for Streamlit Community Cloud free tier. Predictions run live in-memory.",
        }
        return False, info, None

    db, err = get_database_with_error(resolved_uri, resolved_db, timeout_ms=timeout_ms)
    if db is None:
        return False, {"uri": resolved_uri, "database": resolved_db, "status": "Offline", "cloud_mode": False}, err or "Server is unreachable."

    try:
        client = db.client
        server_info = client.server_info()
        collections = db.list_collection_names()

        # Count documents safely
        shipments_count = db[SHIPMENTS_COLLECTION].count_documents({}) if SHIPMENTS_COLLECTION in collections else 0
        predictions_count = db[PREDICTIONS_COLLECTION].count_documents({}) if PREDICTIONS_COLLECTION in collections else 0

        is_atlas = ("mongodb+srv://" in resolved_uri) or ("mongodb.net" in resolved_uri)
        info = {
            "uri": resolved_uri,
            "database": resolved_db,
            "version": server_info.get("version", "unknown"),
            "collections": collections,
            "shipments_count": shipments_count,
            "predictions_count": predictions_count,
            "status": "Connected (Atlas)" if is_atlas else "Connected",
            "cloud_mode": False,
        }
        return True, info, None
    except Exception as e:
        return False, {"uri": resolved_uri, "database": resolved_db, "status": "Offline", "cloud_mode": False}, str(e)


def get_database_with_error(
    uri: Optional[str] = None,
    db_name: Optional[str] = None,
    timeout_ms: int = 4000,
) -> Tuple[Optional[Any], Optional[str]]:
    """Returns the database handle and error message if unavailable."""
    if not PYMONGO_AVAILABLE:
        return None, "pymongo package is not installed."
    uri = get_configured_mongo_uri(uri)
    db_name = get_configured_db_name(db_name)

    if is_mongo_persistence_disabled(uri):
        return None, "MongoDB persistence is gracefully disabled in Streamlit Cloud (no remote MONGODB_URI configured)."

    candidates = get_candidate_uris(uri)
    last_err = None

    for cand in candidates:
        cache_key = (cand, timeout_ms)
        try:
            client = get_mongo_client(cand, timeout_ms=timeout_ms)
            if client is None:
                continue
            client.admin.command("ping")
            target_db = db_name
            try:
                def_db = client.get_default_database()
                if def_db is not None and def_db.name and (not db_name or db_name == DEFAULT_DB_NAME):
                    target_db = def_db.name
            except Exception:
                pass
            return client[target_db], None
        except Exception as e:
            last_err = e
            failed_client = _CLIENTS.pop(cache_key, None)
            if failed_client:
                try:
                    failed_client.close()
                except Exception:
                    pass

    _CLIENTS.pop((uri, timeout_ms), None)

    err_str = str(last_err) if last_err else f"Failed to initialize MongoClient for '{uri}'."
    if "Connection refused" in err_str or "111" in err_str or "10061" in err_str:
        friendly_err = (
            f"Connection to MongoDB on port 27017 refused. The local MongoDB service is currently stopped. "
            f"Start MongoDB via terminal ('net start MongoDB') or Windows Services, then retry."
        )
        return None, friendly_err

    return None, err_str


def get_database(uri: Optional[str] = None, db_name: Optional[str] = None, timeout_ms: int = 4000):
    """Returns the database handle if connected, or None if unavailable."""
    db, _ = get_database_with_error(uri, db_name, timeout_ms=timeout_ms)
    return db


def save_shipment(
    shipment_data: Dict[str, Any],
    uri: Optional[str] = None,
    db_name: Optional[str] = None,
    status: Optional[str] = None,
    timeout_ms: int = 4000,
) -> Dict[str, Any]:
    """
    Saves or updates an actual shipment record in the 'shipments' collection.
    Upserts by shipment_id / Shipment_ID if present.
    """
    uri = get_configured_mongo_uri(uri)
    db_name = get_configured_db_name(db_name)

    if is_mongo_persistence_disabled(uri):
        sid = shipment_data.get("shipment_id") or shipment_data.get("Shipment_ID", "UNKNOWN")
        return {
            "success": False,
            "cloud_mode": True,
            "disabled": True,
            "shipment_id": sid,
            "message": "Streamlit Cloud Mode: MongoDB persistence is disabled for free cloud hosting (no remote MONGODB_URI configured).",
        }

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
    timeout_ms: int = 4000,
) -> Dict[str, Any]:
    """
    Saves a real ML prediction result into the 'risk_predictions' collection.
    Never stores fake data; expects computed ML label, probabilities, and telemetry.
    Supports duplicate prevention via submission_hash matching if prevent_duplicates=True.
    Verifies the connection and actual insert_one() execution.
    """
    uri = get_configured_mongo_uri(uri)
    db_name = get_configured_db_name(db_name)

    if is_mongo_persistence_disabled(uri):
        sid = prediction_data.get("shipment_id") or prediction_data.get("Shipment_ID", "UNKNOWN")
        return {
            "success": False,
            "cloud_mode": True,
            "disabled": True,
            "shipment_id": sid,
            "message": "Streamlit Cloud Mode: MongoDB persistence is disabled for free cloud hosting (no remote MONGODB_URI configured). Prediction was computed live in-memory.",
        }

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
    uri = get_configured_mongo_uri(uri)
    db_name = get_configured_db_name(db_name)
    if is_mongo_persistence_disabled(uri):
        return []

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
    uri = get_configured_mongo_uri(uri)
    db_name = get_configured_db_name(db_name)
    if is_mongo_persistence_disabled(uri):
        return []

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
    uri = get_configured_mongo_uri(uri)
    db_name = get_configured_db_name(db_name)
    if is_mongo_persistence_disabled(uri):
        return {
            "success": False,
            "cloud_mode": True,
            "disabled": True,
            "count": 0,
            "message": "Streamlit Cloud Mode: MongoDB persistence is disabled for free cloud hosting (no remote MONGODB_URI configured).",
        }

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
    timeout_ms: int = 4000,
) -> Dict[str, Any]:
    """
    Bulk saves ML risk prediction records into the 'risk_predictions' collection.
    Verifies database connection and insert_many() execution.
    """
    uri = get_configured_mongo_uri(uri)
    db_name = get_configured_db_name(db_name)

    if is_mongo_persistence_disabled(uri):
        return {
            "success": False,
            "cloud_mode": True,
            "disabled": True,
            "count": 0,
            "message": "Streamlit Cloud Mode: MongoDB persistence is disabled for free cloud hosting (no remote MONGODB_URI configured). Predictions were computed live in-memory.",
        }

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

