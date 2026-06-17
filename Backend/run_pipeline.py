"""
GridLock AI — End-to-End Pipeline Runner
Runs: Clean Data → Feature Engineering → Spillover Risk Model → Impact Scores
Saves all outputs to Backend/outputs/
"""

import sys
import os
import time
import json

# Add Backend to path
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)

from data_pipeline.clean_data import clean_data
from data_pipeline.feature_engineering import engineer_features
from models.zone_clustering import assign_zones, get_zone_summary
from models.spillover_risk_model import train_spillover_model
from models.impact_score import compute_impact_scores


def run_pipeline(data_path: str = None, output_dir: str = None):
    """
    Execute the full GridLock AI pipeline.

    Args:
        data_path: Path to raw CSV file. Defaults to Dataset/ folder.
        output_dir: Output directory. Defaults to Backend/outputs/.
    """
    start_time = time.time()

    print("+" + "=" * 68 + "+")
    print("|" + "  GRIDLOCK AI - PARKING INTELLIGENCE PIPELINE".center(68) + "|")
    print("|" + "  AI-Driven Illegal Parking Detection & Impact Analysis".center(68) + "|")
    print("+" + "=" * 68 + "+")
    print()

    # -- Resolve paths --------------------------------------------------
    if data_path is None:
        data_path = os.path.join(
            BACKEND_DIR, "..", "Dataset",
            "jan to may police violation_anonymized791b166.csv"
        )
    data_path = os.path.abspath(data_path)

    if output_dir is None:
        output_dir = os.path.join(BACKEND_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[PIPELINE] Input:  {data_path}")
    print(f"[PIPELINE] Output: {output_dir}")
    print()

    # -- Step 1: Clean Data ---------------------------------------------
    print("\n" + "-" * 70)
    print("  STEP 1/5: DATA CLEANING")
    print("-" * 70)
    df_clean = clean_data(data_path)

    # -- Step 2: Assign Zones -------------------------------------------
    print("\n" + "-" * 70)
    print("  STEP 2/5: ZONE ASSIGNMENT")
    print("-" * 70)
    df_clean = assign_zones(df_clean)

    # Save cleaned data sample
    clean_sample_path = os.path.join(output_dir, "cleaned_data_sample.csv")
    df_clean.head(1000).to_csv(clean_sample_path, index=False)
    print(f"[PIPELINE] Saved cleaned data sample → {clean_sample_path}")

    # -- Step 3: Feature Engineering ------------------------------------
    print("\n" + "-" * 70)
    print("  STEP 3/5: FEATURE ENGINEERING")
    print("-" * 70)
    zone_features, feature_cols = engineer_features(df_clean)

    # Save zone features
    features_path = os.path.join(output_dir, "zone_features.csv")
    zone_features.to_csv(features_path, index=False)
    print(f"[PIPELINE] Saved zone features → {features_path}")

    # -- Step 4: Train Spillover Risk Model -----------------------------
    print("\n" + "-" * 70)
    print("  STEP 4/5: SPILLOVER RISK MODEL")
    print("-" * 70)
    model, scaler, metrics, zone_features = train_spillover_model(
        zone_features, feature_cols, output_dir=output_dir
    )

    # Save risk scores
    risk_path = os.path.join(output_dir, "zone_risk_scores.csv")
    risk_cols = ["zone_id", "center_lat", "center_lng", "violation_count",
                 "risk_score", "risk_prediction"]
    zone_features[risk_cols].to_csv(risk_path, index=False)
    print(f"[PIPELINE] Saved risk scores → {risk_path}")

    # -- Step 5: Impact Score Calculation -------------------------------
    print("\n" + "-" * 70)
    print("  STEP 5/5: IMPACT SCORE CALCULATION")
    print("-" * 70)
    zone_features = compute_impact_scores(zone_features)

    # Save impact scores
    impact_path = os.path.join(output_dir, "zone_impact_scores.csv")
    impact_cols = [
        "zone_id", "center_lat", "center_lng", "violation_count",
        "risk_score", "impact_score", "severity", "enforcement_priority",
        "impact_risk_component", "impact_density_component",
        "impact_peak_hour_component", "impact_road_impact_component",
        "impact_poi_component", "impact_repeat_offender_component",
    ]
    impact_cols = [c for c in impact_cols if c in zone_features.columns]
    zone_features[impact_cols].to_csv(impact_path, index=False)
    print(f"[PIPELINE] Saved impact scores → {impact_path}")

    # Save hotspot summary (top critical zones)
    hotspot_path = os.path.join(output_dir, "hotspot_summary.csv")
    hotspots = zone_features[zone_features["severity"].isin(["CRITICAL", "HIGH"])].sort_values(
        "enforcement_priority"
    )
    hotspot_export_cols = [
        "zone_id", "center_lat", "center_lng", "violation_count",
        "risk_score", "impact_score", "severity", "enforcement_priority",
        "wrong_parking_ratio", "main_road_parking_ratio",
        "peak_hour_ratio", "repeat_offender_ratio",
        "nearest_metro_dist", "nearest_mall_dist", "poi_count_500m",
    ]
    hotspot_export_cols = [c for c in hotspot_export_cols if c in hotspots.columns]
    hotspots[hotspot_export_cols].to_csv(hotspot_path, index=False)
    print(f"[PIPELINE] Saved hotspot summary → {hotspot_path}")

    # Save full zone data for API
    full_path = os.path.join(output_dir, "zone_full_data.csv")
    zone_features.to_csv(full_path, index=False)
    print(f"[PIPELINE] Saved full zone data → {full_path}")

    # -- Pipeline Summary ----------------------------------------------
    elapsed = time.time() - start_time

    summary = {
        "total_raw_records": int(len(df_clean)),
        "total_zones": int(len(zone_features)),
        "critical_zones": int((zone_features["severity"] == "CRITICAL").sum()),
        "high_zones": int((zone_features["severity"] == "HIGH").sum()),
        "medium_zones": int((zone_features["severity"] == "MEDIUM").sum()),
        "low_zones": int((zone_features["severity"] == "LOW").sum()),
        "model_auc": metrics["auc_roc"],
        "model_f1": metrics["f1"],
        "features_used": len(feature_cols),
        "pipeline_runtime_seconds": round(elapsed, 2),
    }

    summary_path = os.path.join(output_dir, "pipeline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("+" + "=" * 68 + "+")
    print("|" + "  PIPELINE COMPLETE".center(68) + "|")
    print("+" + "=" * 68 + "+")
    print(f"|  Total Violations Processed: {summary['total_raw_records']:>10,}".ljust(69) + "|")
    print(f"|  Total Zones:                {summary['total_zones']:>10,}".ljust(69) + "|")
    print(f"|  Critical Zones:             {summary['critical_zones']:>10,}".ljust(69) + "|")
    print(f"|  High Risk Zones:            {summary['high_zones']:>10,}".ljust(69) + "|")
    print(f"|  Model AUC-ROC:              {summary['model_auc']:>10.4f}".ljust(69) + "|")
    print(f"|  Model F1 Score:             {summary['model_f1']:>10.4f}".ljust(69) + "|")
    print(f"|  Runtime:                    {elapsed:>8.1f}s".ljust(69) + "|")
    print("+" + "=" * 68 + "+")

    return zone_features, metrics, summary


if __name__ == "__main__":
    zone_features, metrics, summary = run_pipeline()
