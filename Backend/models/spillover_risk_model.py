"""
GridLock AI — Spillover Risk Model
XGBoost classifier with self-supervised proxy labels.
Outputs zone → risk_score (0–1).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
import json
import os


def create_proxy_labels(zone_features: pd.DataFrame) -> pd.DataFrame:
    """
    Create self-supervised proxy labels for spillover risk.

    High-risk zones (label=1) meet MULTIPLE conditions:
      - High violation density (above 75th percentile)
      - Significant main-road or crossing parking
      - Proximity to POIs (metro/mall/market)
      - High peak-hour violation ratio

    This multi-criteria approach avoids trivially labeling based on one feature.
    """
    print("[MODEL] Creating proxy labels for spillover risk...")

    # ── Individual risk signals ─────────────────────────────────────────
    # 1. High violation density
    q75_violations = zone_features["violation_count"].quantile(0.75)
    sig_density = (zone_features["violation_count"] >= q75_violations).astype(int)

    # 2. Main road / crossing parking above median
    median_main_road = zone_features["main_road_parking_ratio"].median()
    median_crossing = zone_features["road_crossing_ratio"].median()
    sig_road_impact = (
        (zone_features["main_road_parking_ratio"] > median_main_road) |
        (zone_features["road_crossing_ratio"] > median_crossing)
    ).astype(int)

    # 3. Close to high-traffic POIs
    metro_threshold = zone_features["nearest_metro_dist"].quantile(0.50)
    mall_threshold = zone_features["nearest_mall_dist"].quantile(0.50)
    market_threshold = zone_features["nearest_market_dist"].quantile(0.50)
    sig_poi_proximity = (
        (zone_features["nearest_metro_dist"] < metro_threshold) |
        (zone_features["nearest_mall_dist"] < mall_threshold) |
        (zone_features["nearest_market_dist"] < market_threshold)
    ).astype(int)

    # 4. High peak-hour ratio (congestion happens during peaks)
    median_peak = zone_features["peak_hour_ratio"].median()
    sig_peak_hour = (zone_features["peak_hour_ratio"] > median_peak).astype(int)

    # ── Composite scoring ──────────────────────────────────────────────
    # A zone is "high risk" if it scores ≥ 3 out of 4 signals
    risk_signal_sum = sig_density + sig_road_impact + sig_poi_proximity + sig_peak_hour
    zone_features["spillover_label"] = (risk_signal_sum >= 3).astype(int)

    # Stats
    n_high = zone_features["spillover_label"].sum()
    n_total = len(zone_features)
    print(f"[MODEL] Proxy labels: {n_high:,} high-risk ({n_high/n_total*100:.1f}%) / "
          f"{n_total - n_high:,} low-risk ({(n_total-n_high)/n_total*100:.1f}%)")

    return zone_features


def train_spillover_model(
    zone_features: pd.DataFrame,
    feature_cols: list,
    output_dir: str = None
) -> tuple:
    """
    Train XGBoost spillover risk classifier.

    Args:
        zone_features: Zone-level feature DataFrame with proxy labels.
        feature_cols: List of feature column names.
        output_dir: Directory to save model artifacts.

    Returns:
        (model, scaler, metrics_dict, zone_features_with_predictions)
    """
    print("=" * 70)
    print("  GRIDLOCK AI — SPILLOVER RISK MODEL (XGBoost)")
    print("=" * 70)

    # Create labels
    zone_features = create_proxy_labels(zone_features)

    # Prepare features
    X = zone_features[feature_cols].copy()
    y = zone_features["spillover_label"].values

    # Handle NaN/inf
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n[MODEL] Training set: {len(X_train):,} zones")
    print(f"[MODEL] Test set:     {len(X_test):,} zones")
    print(f"[MODEL] Features:     {len(feature_cols)}")

    # ── XGBoost Classifier ─────────────────────────────────────────────
    # Handle class imbalance with scale_pos_weight
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos = n_neg / max(n_pos, 1)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── Predictions ────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ── Metrics ────────────────────────────────────────────────────────
    metrics = {
        "auc_roc": float(roc_auc_score(y_test, y_prob)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "n_features": len(feature_cols),
        "positive_rate": float(y.mean()),
    }

    print(f"\n[MODEL] ── Evaluation Metrics ──")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")

    print(f"\n[MODEL] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))

    # ── Feature Importance ─────────────────────────────────────────────
    importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
    importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    print(f"[MODEL] Top 10 Feature Importances:")
    for i, (feat, imp) in enumerate(list(importance_sorted.items())[:10]):
        print(f"  {i+1}. {feat}: {imp:.4f}")

    metrics["feature_importance"] = importance_sorted

    # ── Predict risk_score for ALL zones ───────────────────────────────
    X_all_scaled = scaler.transform(X.values)
    zone_features["risk_score"] = model.predict_proba(X_all_scaled)[:, 1]
    zone_features["risk_prediction"] = model.predict(X_all_scaled)

    print(f"\n[MODEL] Risk score distribution:")
    print(f"  Mean:   {zone_features['risk_score'].mean():.4f}")
    print(f"  Median: {zone_features['risk_score'].median():.4f}")
    print(f"  Std:    {zone_features['risk_score'].std():.4f}")
    print(f"  Min:    {zone_features['risk_score'].min():.4f}")
    print(f"  Max:    {zone_features['risk_score'].max():.4f}")

    # ── Save artifacts ─────────────────────────────────────────────────
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir, "spillover_model.joblib")
        scaler_path = os.path.join(output_dir, "feature_scaler.joblib")
        metrics_path = os.path.join(output_dir, "model_metrics.json")

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\n[MODEL] ✅ Saved model to {model_path}")
        print(f"[MODEL] ✅ Saved scaler to {scaler_path}")
        print(f"[MODEL] ✅ Saved metrics to {metrics_path}")

    return model, scaler, metrics, zone_features
