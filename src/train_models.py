"""
SupplyGuard AI - Model Training & Comparison
===============================================
Trains multiple classifiers to predict Risk_Level (Low/Medium/High),
compares them on standard metrics, and persists the best model + artifacts
for the Streamlit dashboard.
"""

import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix, classification_report)
from xgboost import XGBClassifier

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "feature_engineered_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = [
    "Lead_Time_Days", "Distance_KM", "Order_Quantity", "Inventory_Level_Days",
    "Demand_Variability_Pct", "Supplier_Reliability_Score", "Financial_Stability_Score",
    "Geopolitical_Risk_Index", "Weather_Risk_Index", "Port_Congestion_Index",
    "Fuel_Price_Index", "Customs_Delay_Days", "Disruption_Events_Past_Year",
    "On_Time_Delivery_Rate_Pct", "Cost_Variance_Pct", "Single_Sourcing_Flag",
    "Contract_Flexibility_Score", "Geo_Weather_Exposure", "Supplier_Weakness_Score",
    "Logistics_Friction_Score", "Demand_Volatility_Score", "Concentration_Risk_Score",
    "Historical_Instability_Score", "Cost_Pressure_Score",
    "Region_enc", "Product_Category_enc", "Supplier_Tier_enc", "Transport_Mode_enc"
]
TARGET_COL = "Risk_Level"


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS].copy()
    y_raw = df[TARGET_COL].copy()

    le_target = LabelEncoder()
    y = le_target.fit_transform(y_raw)  # High=0? need mapping check
    class_names = list(le_target.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                                  random_state=42, eval_metric="mlogloss", n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
    }

    results = []
    fitted_models = {}
    for name, model in models.items():
        if name in ["Logistic Regression", "KNN", "SVM (RBF)"]:
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
            probs = model.predict_proba(X_test_s)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted")
        rec = recall_score(y_test, preds, average="weighted")
        f1 = f1_score(y_test, preds, average="weighted")
        try:
            auc = roc_auc_score(y_test, probs, multi_class="ovr", average="weighted")
        except Exception:
            auc = np.nan

        results.append({
            "Model": name, "Accuracy": round(acc, 4), "Precision": round(prec, 4),
            "Recall": round(rec, 4), "F1_Score": round(f1, 4), "ROC_AUC": round(auc, 4)
        })
        fitted_models[name] = model

    results_df = pd.DataFrame(results).sort_values("F1_Score", ascending=False).reset_index(drop=True)
    print(results_df)

    best_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    print(f"\nBest model: {best_name}")

    # Confusion matrix + classification report for best model
    if best_name in ["Logistic Regression", "KNN", "SVM (RBF)"]:
        best_preds = best_model.predict(X_test_s)
    else:
        best_preds = best_model.predict(X_test)
    cm = confusion_matrix(y_test, best_preds)
    report = classification_report(y_test, best_preds, target_names=class_names, output_dict=True)

    # Feature importance (best tree-based model, fallback to RF)
    importance_model = fitted_models.get("Random Forest")
    importances = dict(zip(FEATURE_COLS, importance_model.feature_importances_))
    importances = dict(sorted(importances.items(), key=lambda x: -x[1]))

    # Persist artifacts
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(le_target, os.path.join(MODEL_DIR, "target_encoder.pkl"))
    joblib.dump(FEATURE_COLS, os.path.join(MODEL_DIR, "feature_cols.pkl"))
    joblib.dump(fitted_models, os.path.join(MODEL_DIR, "all_models.pkl"))

    # Persist XGBoost model specifically & compute its metrics
    xgb_cm, xgb_report, xgb_importances = None, None, None
    if "XGBoost" in fitted_models:
        xgb_mod = fitted_models["XGBoost"]
        joblib.dump(xgb_mod, os.path.join(MODEL_DIR, "xgboost_model.pkl"))
        xgb_preds = xgb_mod.predict(X_test)
        xgb_cm = confusion_matrix(y_test, xgb_preds).tolist()
        xgb_report = classification_report(y_test, xgb_preds, target_names=class_names, output_dict=True)
        booster = xgb_mod.get_booster()
        score_gain = booster.get_score(importance_type="gain")
        gain_imps = {f: float(score_gain.get(f, 0.0)) for f in FEATURE_COLS}
        tot_g = sum(gain_imps.values()) or 1.0
        xgb_importances = {k: round(v / tot_g, 4) for k, v in sorted(gain_imps.items(), key=lambda x: -x[1])}

    comparison_data = {
        "results": results_df.to_dict(orient="records"),
        "best_model": best_name,
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "feature_importances": importances,
    }
    if xgb_cm:
        comparison_data["xgboost_confusion_matrix"] = xgb_cm
        comparison_data["xgboost_classification_report"] = xgb_report
        comparison_data["xgboost_feature_importances"] = xgb_importances

    with open(os.path.join(MODEL_DIR, "model_comparison.json"), "w") as f:
        json.dump(comparison_data, f, indent=2)

    results_df.to_csv(os.path.join(MODEL_DIR, "model_comparison.csv"), index=False)
    print("\nArtifacts saved to", MODEL_DIR)


if __name__ == "__main__":
    main()
