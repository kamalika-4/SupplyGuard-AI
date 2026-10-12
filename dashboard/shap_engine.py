"""
SupplyGuard AI - SHAP Explainable AI Engine
============================================
Computes exact Shapley values using XGBoost's native TreeSHAP implementation
and generates interactive Plotly waterfall, beeswarm, and dependence plots
for both local (single-shipment) and global (network-wide) model explanations.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xgboost as xgb
import shap

PALETTE = {
    "bg": "#0B1220",
    "panel": "#111B2E",
    "panel2": "#16223A",
    "grid": "#233350",
    "text": "#E7EDF7",
    "muted": "#8CA0C4",
    "safe": "#2DD4BF",
    "watch": "#FFC94D",
    "elevated": "#FF9F43",
    "critical": "#FF5A5F",
    "accent": "#5B8CFF",
}

FRIENDLY_NAMES = {
    "Geo_Weather_Exposure": "Geopolitical & Weather Exposure",
    "Supplier_Weakness_Score": "Supplier Weakness Score",
    "Logistics_Friction_Score": "Logistics Friction Score",
    "Demand_Volatility_Score": "Demand Volatility Score",
    "Concentration_Risk_Score": "Concentration Risk Score",
    "Historical_Instability_Score": "Historical Instability Score",
    "Cost_Pressure_Score": "Cost Pressure Score",
    "Geopolitical_Risk_Index": "Geopolitical Risk Index",
    "Weather_Risk_Index": "Weather Risk Index",
    "Port_Congestion_Index": "Port Congestion Index",
    "Fuel_Price_Index": "Fuel Price Index",
    "Supplier_Reliability_Score": "Supplier Reliability Score",
    "Financial_Stability_Score": "Financial Stability Score",
    "Customs_Delay_Days": "Customs Delay (days)",
    "Disruption_Events_Past_Year": "Disruption Events (past year)",
    "On_Time_Delivery_Rate_Pct": "On-Time Delivery Rate (%)",
    "Cost_Variance_Pct": "Cost Variance (%)",
    "Single_Sourcing_Flag": "Single Sourcing Flag",
    "Contract_Flexibility_Score": "Contract Flexibility Score",
    "Lead_Time_Days": "Lead Time (days)",
    "Distance_KM": "Distance (km)",
    "Order_Quantity": "Order Quantity",
    "Inventory_Level_Days": "Inventory Level (days)",
    "Demand_Variability_Pct": "Demand Variability (%)",
    "Region_enc": "Region (encoded)",
    "Product_Category_enc": "Product Category (encoded)",
    "Supplier_Tier_enc": "Supplier Tier (encoded)",
    "Transport_Mode_enc": "Transport Mode (encoded)",
}


def compute_local_shap(
    xgb_model,
    sample_encoded: dict,
    feature_cols: list,
    target_class: str = "High",
    target_encoder=None,
) -> dict:
    """Computes exact Shapley contributions for a single shipment using the actual
    trained XGBoost model and the shipment's real feature values.
    """
    booster = xgb_model.get_booster()
    X_df = pd.DataFrame([sample_encoded])[feature_cols]
    dmat = xgb.DMatrix(X_df)

    # 1. Exact TreeSHAP contributions from actual model (shape: num_classes, num_features + 1)
    contribs = booster.predict(dmat, pred_contribs=True)[0]
    # 2. Exact raw model margins on log-odds scale (shape: num_classes)
    raw_margins = booster.predict(dmat, output_margin=True)[0]
    # 3. Exact multi-class predicted probabilities (shape: num_classes)
    probabilities = booster.predict(dmat)[0]

    class_names = list(target_encoder.classes_) if target_encoder else ["High", "Low", "Medium"]
    class_idx = 0
    if target_class in class_names:
        class_idx = class_names.index(target_class)

    shap_vals = contribs[class_idx, :-1]
    base_val = float(contribs[class_idx, -1])
    model_output_log_odds = float(raw_margins[class_idx])
    disruption_prob = float(probabilities[class_idx])
    sigmoid_prob = float(1.0 / (1.0 + np.exp(-model_output_log_odds)))
    baseline_prob = float(1.0 / (1.0 + np.exp(-base_val)))

    feature_shap = dict(zip(feature_cols, [float(v) for v in shap_vals]))

    # Rich item breakdown with actual feature values
    items = []
    for f in feature_cols:
        val = float(feature_shap[f])
        raw_val = sample_encoded.get(f, 0.0)
        items.append({
            "feature": f,
            "friendly_name": FRIENDLY_NAMES.get(f, f),
            "shap_value": val,
            "raw_value": raw_val,
            "abs_shap": abs(val),
        })

    # Sort items by magnitude of attribution
    sorted_items = sorted(items, key=lambda x: x["abs_shap"], reverse=True)
    risk_drivers = [x for x in sorted_items if x["shap_value"] > 0]
    protective_factors = [x for x in sorted_items if x["shap_value"] < 0]

    sum_shap = float(np.sum(shap_vals))
    sum_positive = float(sum([x["shap_value"] for x in risk_drivers]))
    sum_negative = float(sum([x["shap_value"] for x in protective_factors]))

    return {
        "target_class": target_class,
        "class_names": class_names,
        "feature_shap": feature_shap,
        "base_value": base_val,
        "margin_value": model_output_log_odds,
        "sum_shap": sum_shap,
        "sum_positive": sum_positive,
        "sum_negative": sum_negative,
        "disruption_prob": disruption_prob,
        "sigmoid_prob": sigmoid_prob,
        "baseline_prob": baseline_prob,
        "probabilities": {cls: float(p) for cls, p in zip(class_names, probabilities)},
        "sorted_items": sorted_items,
        "risk_drivers": risk_drivers,
        "protective_factors": protective_factors,
        "raw_features": {f: sample_encoded.get(f, 0.0) for f in feature_cols},
    }


def plot_shap_waterfall(
    local_shap_result: dict,
    max_display: int = 8,
    target_class: str = "High",
    height: int = 420,
) -> go.Figure:
    """Creates an interactive Plotly Waterfall plot showing how individual features
    push model predictions toward or away from disruption risk on the exact log-odds scale."""
    sorted_items = local_shap_result["sorted_items"]
    base_val = local_shap_result["base_value"]
    model_output = local_shap_result["margin_value"]
    disruption_prob = local_shap_result["disruption_prob"]

    top_items = sorted_items[:max_display]
    other_val = float(sum([item["shap_value"] for item in sorted_items[max_display:]]))

    measures = ["absolute"]
    y_labels = ["Base Value E[f(X)]"]
    x_values = [base_val]
    text_labels = [f"{base_val:+.2f}"]
    hover_texts = [
        f"<b>Portfolio Baseline</b><br>"
        f"Base Log-Odds: {base_val:+.4f}<br>"
        f"Implied Baseline Risk: {local_shap_result['baseline_prob']*100:.1f}%"
    ]

    running_margin = base_val

    for item in top_items:
        val = item["shap_value"]
        running_margin += val
        raw_val = item["raw_value"]
        if isinstance(raw_val, float):
            formatted_raw = f"{raw_val:.2f}"
        elif isinstance(raw_val, (int, np.integer)):
            formatted_raw = f"{raw_val:,}"
        else:
            formatted_raw = str(raw_val)

        measures.append("relative")
        label = f"{item['friendly_name']} = {formatted_raw}"
        y_labels.append(label)
        x_values.append(val)
        text_labels.append(f"{val:+.2f}")

        running_prob = float(1.0 / (1.0 + np.exp(-running_margin)))
        push_type = "Elevating Risk" if val > 0 else "Mitigating Risk"
        hover_texts.append(
            f"<b>{item['friendly_name']}</b><br>"
            f"Raw Value: {formatted_raw}<br>"
            f"Attribution: {val:+.4f} log-odds ({push_type})<br>"
            f"Cumulative Running Margin: {running_margin:+.4f}<br>"
            f"Running Implied Probability: {running_prob*100:.1f}%"
        )

    if len(sorted_items) > max_display:
        running_margin += other_val
        measures.append("relative")
        y_labels.append(f"Other ({len(sorted_items) - max_display} features)")
        x_values.append(other_val)
        text_labels.append(f"{other_val:+.2f}")
        running_prob = float(1.0 / (1.0 + np.exp(-running_margin)))
        hover_texts.append(
            f"<b>Remaining {len(sorted_items) - max_display} Features</b><br>"
            f"Combined Attribution: {other_val:+.4f} log-odds<br>"
            f"Cumulative Running Margin: {running_margin:+.4f}<br>"
            f"Running Implied Probability: {running_prob*100:.1f}%"
        )

    measures.append("total")
    y_labels.append(f"Model Output f(X)")
    x_values.append(model_output)
    text_labels.append(f"{model_output:+.2f} ({disruption_prob*100:.1f}%)")
    hover_texts.append(
        f"<b>Final Model Output ({target_class} Risk)</b><br>"
        f"Model Output Log-Odds Margin: <b>{model_output:+.4f}</b><br>"
        f"Disruption Risk Probability: <b>{disruption_prob*100:.2f}%</b><br>"
        f"Identity: Base ({base_val:+.4f}) + &Sigma;SHAP ({local_shap_result['sum_shap']:+.4f}) = {model_output:+.4f}"
    )

    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            measure=measures,
            y=y_labels,
            x=x_values,
            text=text_labels,
            textposition="outside",
            hoverinfo="text",
            hovertext=hover_texts,
            decreasing=dict(marker=dict(color=PALETTE["safe"])),
            increasing=dict(marker=dict(color=PALETTE["critical"])),
            totals=dict(marker=dict(color=PALETTE["accent"])),
            connector=dict(line=dict(color=PALETTE["grid"], width=1.5)),
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text"], family="Inter"),
        height=height,
        margin=dict(l=10, r=40, t=30, b=40),
        xaxis=dict(
            title=f"Log-Odds Margin Scale ({target_class} Disruption Risk)",
            gridcolor=PALETTE["grid"],
            zerolinecolor=PALETTE["muted"],
            zerolinewidth=1.5,
        ),
        yaxis=dict(
            autorange="reversed",
            gridcolor=PALETTE["grid"],
        ),
        showlegend=False,
    )
    return fig


def compute_global_shap_data(
    xgb_model,
    df: pd.DataFrame,
    feature_cols: list,
    target_encoder=None,
    sample_size: int = 400,
) -> dict:
    """Computes global SHAP values over a sampled evaluation subset for quick dashboard rendering."""
    sample = df.sample(min(sample_size, len(df)), random_state=42)
    X_sample = sample[feature_cols].copy()
    booster = xgb_model.get_booster()
    dmat = xgb.DMatrix(X_sample)
    contribs = booster.predict(dmat, pred_contribs=True)  # (N, num_classes, num_features + 1)

    class_names = list(target_encoder.classes_) if target_encoder else ["High", "Low", "Medium"]
    high_idx = class_names.index("High") if "High" in class_names else 0

    # Mean absolute SHAP per feature across all classes
    mean_abs_all = np.mean(np.abs(contribs[:, :, :-1]), axis=(0, 1))
    ranking = [f for _, f in sorted(zip(mean_abs_all, feature_cols), reverse=True)]

    return {
        "X_sample": X_sample,
        "contribs": contribs,
        "feature_cols": feature_cols,
        "class_names": class_names,
        "high_idx": high_idx,
        "ranking": ranking,
        "mean_abs_all": dict(zip(feature_cols, mean_abs_all)),
    }


def plot_shap_summary_bar(
    global_shap_data: dict,
    top_n: int = 15,
    height: int = 460,
) -> go.Figure:
    """Plots global feature importance measured by mean absolute SHAP value across classes."""
    contribs = global_shap_data["contribs"]
    feature_cols = global_shap_data["feature_cols"]
    class_names = global_shap_data["class_names"]

    fig = go.Figure()
    class_colors = {
        "High": PALETTE["critical"],
        "Medium": PALETTE["watch"],
        "Low": PALETTE["safe"],
    }

    # Order features by overall mean absolute SHAP
    top_feats = global_shap_data["ranking"][:top_n][::-1]

    for c_idx, c_name in enumerate(class_names):
        vals = []
        for feat in top_feats:
            f_idx = feature_cols.index(feat)
            mean_abs = float(np.mean(np.abs(contribs[:, c_idx, f_idx])))
            vals.append(mean_abs)

        friendly_labels = [FRIENDLY_NAMES.get(f, f) for f in top_feats]
        fig.add_trace(
            go.Bar(
                y=friendly_labels,
                x=vals,
                name=f"{c_name} Risk Impact",
                orientation="h",
                marker=dict(color=class_colors.get(c_name, PALETTE["accent"])),
            )
        )

    fig.update_layout(
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text"], family="Inter"),
        height=height,
        margin=dict(l=10, r=10, t=30, b=30),
        xaxis=dict(title="Mean |SHAP Value| (Impact on Model Magnitude)", gridcolor=PALETTE["grid"]),
        yaxis=dict(gridcolor=PALETTE["grid"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_shap_beeswarm_interactive(
    global_shap_data: dict,
    target_class: str = "High",
    top_n: int = 12,
    height: int = 480,
) -> go.Figure:
    """Generates an interactive Plotly Beeswarm-style distribution plot showing how high vs
    low feature values push predictions toward or away from the target risk level."""
    contribs = global_shap_data["contribs"]
    X_sample = global_shap_data["X_sample"]
    feature_cols = global_shap_data["feature_cols"]
    class_names = global_shap_data["class_names"]
    c_idx = class_names.index(target_class) if target_class in class_names else 0

    top_feats = global_shap_data["ranking"][:top_n][::-1]

    fig = go.Figure()
    np.random.seed(42)

    for feat_idx_in_top, feat in enumerate(top_feats):
        f_idx = feature_cols.index(feat)
        shap_vals = contribs[:, c_idx, f_idx]
        raw_vals = X_sample[feat].values

        # Normalize raw values 0-1 for color scale
        min_v, max_v = raw_vals.min(), raw_vals.max()
        norm_vals = (raw_vals - min_v) / (max_v - min_v + 1e-8)

        # Subtle vertical jitter so points don't stack directly
        jitter = np.random.uniform(-0.18, 0.18, size=len(shap_vals))
        y_positions = feat_idx_in_top + jitter

        hover_text = [
            f"<b>{FRIENDLY_NAMES.get(feat, feat)}</b><br>"
            f"Raw Value: {rv:.2f}<br>"
            f"SHAP Impact ({target_class}): {sv:+.3f}"
            for rv, sv in zip(raw_vals, shap_vals)
        ]

        fig.add_trace(
            go.Scatter(
                x=shap_vals,
                y=y_positions,
                mode="markers",
                marker=dict(
                    size=5.5,
                    color=norm_vals,
                    colorscale=[
                        [0.0, "#2DD4BF"],   # Low feature value: teal/blue
                        [0.5, "#FFC94D"],   # Mid feature value: yellow
                        [1.0, "#FF5A5F"],   # High feature value: red
                    ],
                    opacity=0.75,
                    showscale=(feat_idx_in_top == len(top_feats) - 1),
                    colorbar=dict(
                        title=dict(text="Feature Value", side="right"),
                        tickvals=[0, 1],
                        ticktext=["Low", "High"],
                        len=0.5,
                        y=0.5,
                    ) if (feat_idx_in_top == len(top_feats) - 1) else None,
                ),
                text=hover_text,
                hoverinfo="text",
                showlegend=False,
            )
        )

    # Reference zero line
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color=PALETTE["muted"])

    friendly_labels = [FRIENDLY_NAMES.get(f, f) for f in top_feats]
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text"], family="Inter"),
        height=height,
        margin=dict(l=10, r=20, t=30, b=40),
        xaxis=dict(
            title=f"SHAP Value (Impact on Predicting {target_class} Risk)",
            gridcolor=PALETTE["grid"],
            zerolinecolor=PALETTE["grid"],
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(top_feats))),
            ticktext=friendly_labels,
            gridcolor=PALETTE["grid"],
        ),
    )
    return fig


def plot_shap_dependence_interactive(
    global_shap_data: dict,
    feature_name: str,
    target_class: str = "High",
    height: int = 400,
) -> go.Figure:
    """Generates an interactive SHAP dependence plot for a chosen feature."""
    contribs = global_shap_data["contribs"]
    X_sample = global_shap_data["X_sample"]
    feature_cols = global_shap_data["feature_cols"]
    class_names = global_shap_data["class_names"]
    c_idx = class_names.index(target_class) if target_class in class_names else 0

    if feature_name not in feature_cols:
        feature_name = feature_cols[0]

    f_idx = feature_cols.index(feature_name)
    x_vals = X_sample[feature_name].values
    shap_vals = contribs[:, c_idx, f_idx]

    # Use another correlated feature for interaction color (e.g. Early Warning Score or Geo Weather)
    color_col = "Geo_Weather_Exposure" if feature_name != "Geo_Weather_Exposure" else "Supplier_Weakness_Score"
    color_vals = X_sample[color_col].values if color_col in X_sample.columns else x_vals

    fig = go.Figure(
        go.Scatter(
            x=x_vals,
            y=shap_vals,
            mode="markers",
            marker=dict(
                size=7,
                color=color_vals,
                colorscale="Viridis",
                colorbar=dict(title=FRIENDLY_NAMES.get(color_col, color_col), len=0.6),
                opacity=0.85,
            ),
            text=[
                f"{FRIENDLY_NAMES.get(feature_name, feature_name)}: {x:.2f}<br>"
                f"SHAP: {s:+.3f}<br>"
                f"{FRIENDLY_NAMES.get(color_col, color_col)}: {c:.2f}"
                for x, s, c in zip(x_vals, shap_vals, color_vals)
            ],
            hoverinfo="text",
        )
    )

    fig.add_hline(y=0, line_dash="dash", line_color=PALETTE["muted"])

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text"], family="Inter"),
        height=height,
        margin=dict(l=10, r=10, t=30, b=30),
        xaxis=dict(title=FRIENDLY_NAMES.get(feature_name, feature_name), gridcolor=PALETTE["grid"]),
        yaxis=dict(title=f"SHAP Value ({target_class} Risk)", gridcolor=PALETTE["grid"]),
    )
    return fig
