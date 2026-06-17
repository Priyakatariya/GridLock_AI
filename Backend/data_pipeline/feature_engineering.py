"""
GridLock AI — Feature Engineering
Extracts zone-level features from cleaned violation data for the spillover risk model.
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data_pipeline.poi_data import BENGALURU_POIS, get_pois_by_category


def haversine_distance(lat1, lng1, lat2, lng2):
    """
    Calculate the Haversine distance in meters between two points.
    Vectorized version for numpy arrays.
    """
    R = 6371000  # Earth's radius in meters

    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlng = np.radians(lng2 - lng1)

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlng / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def compute_poi_features(zone_centers: pd.DataFrame) -> pd.DataFrame:
    """
    Compute POI proximity features for each zone.

    Args:
        zone_centers: DataFrame with zone_id, center_lat, center_lng

    Returns:
        DataFrame with POI distance and count features per zone.
    """
    print("[FEAT] Computing POI proximity features...")

    poi_categories = {
        "metro_station": "nearest_metro_dist",
        "mall": "nearest_mall_dist",
        "hospital": "nearest_hospital_dist",
        "market": "nearest_market_dist",
        "bus_stand": "nearest_bus_stand_dist",
        "school": "nearest_school_dist",
        "event_venue": "nearest_event_venue_dist",
    }

    zone_lats = zone_centers["center_lat"].values
    zone_lngs = zone_centers["center_lng"].values

    for category, col_name in poi_categories.items():
        pois = get_pois_by_category(category)
        if not pois:
            zone_centers[col_name] = np.nan
            continue

        poi_lats = np.array([p[1] for p in pois])
        poi_lngs = np.array([p[2] for p in pois])

        # Compute distance from each zone center to each POI in this category
        min_dists = []
        for zlat, zlng in zip(zone_lats, zone_lngs):
            dists = haversine_distance(zlat, zlng, poi_lats, poi_lngs)
            min_dists.append(np.min(dists))

        zone_centers[col_name] = min_dists

    # Count POIs within 500m radius
    all_poi_lats = np.array([p[1] for p in BENGALURU_POIS])
    all_poi_lngs = np.array([p[2] for p in BENGALURU_POIS])

    poi_counts = []
    for zlat, zlng in zip(zone_lats, zone_lngs):
        dists = haversine_distance(zlat, zlng, all_poi_lats, all_poi_lngs)
        poi_counts.append(np.sum(dists <= 500))

    zone_centers["poi_count_500m"] = poi_counts

    # Also count within 1km
    poi_counts_1km = []
    for zlat, zlng in zip(zone_lats, zone_lngs):
        dists = haversine_distance(zlat, zlng, all_poi_lats, all_poi_lngs)
        poi_counts_1km.append(np.sum(dists <= 1000))

    zone_centers["poi_count_1km"] = poi_counts_1km

    print(f"[FEAT] ✅ POI features computed for {len(zone_centers):,} zones")
    return zone_centers


def compute_violation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute zone-level violation features.

    Args:
        df: Cleaned DataFrame with zone_id assigned.

    Returns:
        DataFrame with one row per zone and aggregated features.
    """
    print("[FEAT] Computing zone-level violation features...")

    zone_features = df.groupby("zone_id").agg(
        # Core counts
        violation_count=("id", "count"),
        center_lat=("zone_center_lat", "first"),
        center_lng=("zone_center_lng", "first"),

        # Violation type ratios
        wrong_parking_count=("is_wrong_parking", "sum"),
        no_parking_count=("is_no_parking", "sum"),
        main_road_parking_count=("is_main_road_parking", "sum"),
        road_crossing_parking_count=("is_road_crossing_parking", "sum"),
        footpath_parking_count=("is_footpath_parking", "sum"),
        double_parking_count=("is_double_parking", "sum"),

        # Temporal features
        peak_hour_count=("is_peak_hour", "sum"),
        weekend_count=("is_weekend", "sum"),

        # Junction
        junction_violation_count=("is_near_junction", "sum"),

        # Unique vehicles (for repeat offender detection)
        total_vehicles=("vehicle_number", "count"),
        unique_vehicles=("vehicle_number", "nunique"),

        # Vehicle type diversity
        unique_vehicle_types=("vehicle_type_clean", "nunique"),

        # Unique violation types seen in this zone
        unique_violation_types=("violation_count_per_record", lambda x: x.sum()),

        # Temporal spread
        active_hours=("hour", "nunique"),
        active_days=("day_of_week", "nunique"),

        # Response time
        avg_response_time=("response_time_hours", "mean"),
        median_response_time=("response_time_hours", "median"),

        # Police stations covering this zone
        police_stations=("police_station", "nunique"),
    ).reset_index()

    # ── Derived ratios ──────────────────────────────────────────────────
    n = zone_features["violation_count"]

    zone_features["wrong_parking_ratio"] = zone_features["wrong_parking_count"] / n
    zone_features["no_parking_ratio"] = zone_features["no_parking_count"] / n
    zone_features["main_road_parking_ratio"] = zone_features["main_road_parking_count"] / n
    zone_features["road_crossing_ratio"] = zone_features["road_crossing_parking_count"] / n
    zone_features["footpath_parking_ratio"] = zone_features["footpath_parking_count"] / n
    zone_features["double_parking_ratio"] = zone_features["double_parking_count"] / n
    zone_features["peak_hour_ratio"] = zone_features["peak_hour_count"] / n
    zone_features["weekend_ratio"] = zone_features["weekend_count"] / n
    zone_features["junction_violation_ratio"] = zone_features["junction_violation_count"] / n

    # Repeat offender ratio: if many violations come from fewer unique vehicles
    zone_features["repeat_offender_ratio"] = 1 - (
        zone_features["unique_vehicles"] / zone_features["total_vehicles"]
    )
    zone_features["repeat_offender_ratio"] = zone_features["repeat_offender_ratio"].clip(0, 1)

    # Vehicle diversity (normalized)
    max_vtype = zone_features["unique_vehicle_types"].max()
    zone_features["vehicle_diversity"] = zone_features["unique_vehicle_types"] / max(max_vtype, 1)

    # Temporal coverage: how many of 24 hours and 7 days have violations
    zone_features["temporal_spread"] = (
        (zone_features["active_hours"] / 24) * 0.5 +
        (zone_features["active_days"] / 7) * 0.5
    )

    # Fill NaN response times with median
    median_response = zone_features["avg_response_time"].median()
    zone_features["avg_response_time"] = zone_features["avg_response_time"].fillna(median_response)
    zone_features["median_response_time"] = zone_features["median_response_time"].fillna(median_response)

    print(f"[FEAT] ✅ Violation features computed for {len(zone_features):,} zones")
    return zone_features


def compute_density_features(zone_features: pd.DataFrame) -> pd.DataFrame:
    """
    Compute violation density features (relative to other zones).
    """
    print("[FEAT] Computing density features...")

    # Log-normalized violation density
    zone_features["log_violation_count"] = np.log1p(zone_features["violation_count"])

    # Percentile rank within all zones
    zone_features["violation_density_rank"] = (
        zone_features["violation_count"].rank(pct=True)
    )

    # Z-score normalization
    mean_v = zone_features["violation_count"].mean()
    std_v = zone_features["violation_count"].std()
    zone_features["violation_zscore"] = (zone_features["violation_count"] - mean_v) / max(std_v, 1)

    print(f"[FEAT] ✅ Density features computed")
    return zone_features


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Args:
        df: Cleaned DataFrame with zone_id assigned.

    Returns:
        Zone-level feature DataFrame ready for modeling.
    """
    print("=" * 70)
    print("  GRIDLOCK AI — FEATURE ENGINEERING")
    print("=" * 70)

    # Step 1: Aggregate violation features per zone
    zone_features = compute_violation_features(df)

    # Step 2: Add POI proximity features
    zone_features = compute_poi_features(zone_features)

    # Step 3: Add density features
    zone_features = compute_density_features(zone_features)

    # Final feature list for modeling
    feature_cols = [
        "violation_count", "log_violation_count", "violation_density_rank",
        "wrong_parking_ratio", "no_parking_ratio", "main_road_parking_ratio",
        "road_crossing_ratio", "footpath_parking_ratio", "double_parking_ratio",
        "peak_hour_ratio", "weekend_ratio",
        "junction_violation_ratio", "repeat_offender_ratio",
        "vehicle_diversity", "temporal_spread",
        "avg_response_time",
        "nearest_metro_dist", "nearest_mall_dist", "nearest_hospital_dist",
        "nearest_market_dist", "nearest_bus_stand_dist",
        "poi_count_500m", "poi_count_1km",
    ]

    # Keep only features that exist
    feature_cols = [c for c in feature_cols if c in zone_features.columns]

    print(f"\n[FEAT] ✅ Feature engineering complete!")
    print(f"[FEAT] Zones: {len(zone_features):,}")
    print(f"[FEAT] Features ({len(feature_cols)}): {feature_cols}")

    return zone_features, feature_cols
