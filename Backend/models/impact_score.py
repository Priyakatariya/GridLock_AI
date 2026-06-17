"""
GridLock AI — Impact Score Calculation
Computes a weighted impact score per zone combining risk score, violation
density, temporal patterns, road type, and POI proximity.
"""

import pandas as pd
import numpy as np


def normalize_column(series: pd.Series) -> pd.Series:
    """Min-max normalize a column to [0, 1]."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)
    return (series - min_val) / (max_val - min_val)


def compute_poi_proximity_score(zone_features: pd.DataFrame) -> pd.Series:
    """
    Compute a composite POI proximity score.
    Closer to POIs → higher score (inverted distance).
    """
    # Take the nearest distance among key POI types
    distance_cols = [
        "nearest_metro_dist", "nearest_mall_dist",
        "nearest_market_dist", "nearest_bus_stand_dist",
    ]

    existing_cols = [c for c in distance_cols if c in zone_features.columns]
    if not existing_cols:
        return pd.Series(0.5, index=zone_features.index)

    # Average nearest distance across POI types
    avg_dist = zone_features[existing_cols].mean(axis=1)

    # Invert and normalize (closer = higher score)
    max_dist = avg_dist.max()
    if max_dist == 0:
        return pd.Series(0.5, index=zone_features.index)

    proximity_score = 1 - (avg_dist / max_dist)
    return proximity_score.clip(0, 1)


def compute_violation_density_score(zone_features: pd.DataFrame) -> pd.Series:
    """
    Compute violation density score using log-normalized counts.
    """
    log_count = np.log1p(zone_features["violation_count"])
    return normalize_column(log_count)


def compute_impact_scores(zone_features: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the final impact score for each zone.

    Impact Score = weighted combination of:
        - Spillover risk score (from ML model)      : 0.30
        - Violation density                          : 0.20
        - Peak hour violation ratio                  : 0.15
        - Main road / crossing parking ratio         : 0.15
        - POI proximity                              : 0.10
        - Repeat offender ratio                      : 0.10

    Args:
        zone_features: DataFrame with risk_score and all engineered features.

    Returns:
        DataFrame with impact_score and severity classification.
    """
    print("=" * 70)
    print("  GRIDLOCK AI — IMPACT SCORE CALCULATION")
    print("=" * 70)

    # ── Component scores (all normalized to 0–1) ──────────────────────
    components = {}

    # 1. Spillover risk score (already 0–1 from model)
    components["risk_component"] = zone_features["risk_score"].clip(0, 1)

    # 2. Violation density
    components["density_component"] = compute_violation_density_score(zone_features)

    # 3. Peak hour ratio (already 0–1)
    components["peak_hour_component"] = zone_features["peak_hour_ratio"].clip(0, 1)

    # 4. Road impact (main road + crossing parking)
    road_impact = (
        zone_features["main_road_parking_ratio"] * 0.6 +
        zone_features["road_crossing_ratio"] * 0.4
    )
    components["road_impact_component"] = road_impact.clip(0, 1)

    # 5. POI proximity
    components["poi_component"] = compute_poi_proximity_score(zone_features)

    # 6. Repeat offender ratio (already 0–1)
    components["repeat_offender_component"] = zone_features["repeat_offender_ratio"].clip(0, 1)

    # ── Weighted combination ──────────────────────────────────────────
    weights = {
        "risk_component": 0.30,
        "density_component": 0.20,
        "peak_hour_component": 0.15,
        "road_impact_component": 0.15,
        "poi_component": 0.10,
        "repeat_offender_component": 0.10,
    }

    # Compute weighted sum
    impact_score = sum(
        components[name] * weight for name, weight in weights.items()
    )

    zone_features["impact_score"] = impact_score.clip(0, 1)

    # Store individual components for transparency
    for name, values in components.items():
        zone_features[f"impact_{name}"] = values

    # ── Severity classification (percentile-based) ──────────────────
    # Use percentiles so the classification adapts to actual data distribution
    p95 = zone_features["impact_score"].quantile(0.95)
    p80 = zone_features["impact_score"].quantile(0.80)
    p50 = zone_features["impact_score"].quantile(0.50)

    def classify_severity(score):
        if score >= p95:
            return "CRITICAL"
        elif score >= p80:
            return "HIGH"
        elif score >= p50:
            return "MEDIUM"
        else:
            return "LOW"

    zone_features["severity"] = zone_features["impact_score"].apply(classify_severity)

    print(f"\n[IMPACT] Severity Thresholds (percentile-based):")
    print(f"  CRITICAL: >= {p95:.4f} (top 5%)")
    print(f"  HIGH:     >= {p80:.4f} (top 20%)")
    print(f"  MEDIUM:   >= {p50:.4f} (top 50%)")
    print(f"  LOW:      < {p50:.4f}")

    # ── Priority rank ─────────────────────────────────────────────────
    zone_features["enforcement_priority"] = (
        zone_features["impact_score"].rank(ascending=False, method="dense").astype(int)
    )

    # ── Summary statistics ────────────────────────────────────────────
    print(f"\n[IMPACT] Impact Score Distribution:")
    print(f"  Mean:   {zone_features['impact_score'].mean():.4f}")
    print(f"  Median: {zone_features['impact_score'].median():.4f}")
    print(f"  Std:    {zone_features['impact_score'].std():.4f}")
    print(f"  Min:    {zone_features['impact_score'].min():.4f}")
    print(f"  Max:    {zone_features['impact_score'].max():.4f}")

    severity_counts = zone_features["severity"].value_counts()
    print(f"\n[IMPACT] Severity Distribution:")
    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        count = severity_counts.get(sev, 0)
        pct = count / len(zone_features) * 100
        print(f"  {sev:10s}: {count:5,} zones ({pct:.1f}%)")

    print(f"\n[IMPACT] Component Weights: {weights}")

    # Top-10 hotspots
    top10 = zone_features.nsmallest(10, "enforcement_priority")[
        ["zone_id", "center_lat", "center_lng", "violation_count",
         "risk_score", "impact_score", "severity", "enforcement_priority"]
    ]
    print(f"\n[IMPACT] ── Top 10 Enforcement Priority Zones ──")
    print(top10.to_string(index=False))

    print(f"\n[IMPACT] ✅ Impact scores computed for {len(zone_features):,} zones")

    return zone_features
