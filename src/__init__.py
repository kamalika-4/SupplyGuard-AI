"""
SupplyGuard AI - Source Package
================================
Exports core utilities and database integration modules for seamless imports
both locally and under Streamlit Community Cloud.
"""

try:
    from . import mongo_db
except ImportError:
    import mongo_db

from .mongo_db import (
    DEFAULT_MONGO_URI,
    DEFAULT_DB_NAME,
    SHIPMENTS_COLLECTION,
    PREDICTIONS_COLLECTION,
    PYMONGO_AVAILABLE,
    get_configured_mongo_uri,
    get_configured_db_name,
    is_cloud_environment,
    is_mongo_persistence_disabled,
    sanitize_for_mongo,
    get_candidate_uris,
    get_mongo_client,
    close_mongo_clients,
    check_mongo_connection,
    get_database_with_error,
    get_database,
    save_shipment,
    save_risk_prediction,
    get_shipments,
    get_risk_predictions,
    bulk_save_shipments,
    bulk_save_risk_predictions,
)

__all__ = [
    "mongo_db",
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
