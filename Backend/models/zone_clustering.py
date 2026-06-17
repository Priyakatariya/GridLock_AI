"""
GridLock AI — Zone Clustering
Divides Bengaluru into a grid of ~500m × 500m zones.
Each violation is assigned a zone_id based on its lat/lng.
"""

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# At Bengaluru's latitude (~13°N):
#   1° latitude  ≈ 111,000 meters
#   1° longitude ≈ 108,000 meters (cos(13°) × 111,000)
#
# For 500m grid cells:
#   Δlat = 500 / 111,000 ≈ 0.0045
#   Δlng = 500 / 108,000 ≈ 0.0046
# ─────────────────────────────────────────────────────────────────────────────

GRID_SIZE_METERS = 500
LAT_STEP = GRID_SIZE_METERS / 111_000       # ~0.0045 degrees
LNG_STEP = GRID_SIZE_METERS / 108_000       # ~0.0046 degrees

# Bengaluru bounding box
BENGALURU_LAT_MIN = 12.7
BENGALURU_LAT_MAX = 13.21
BENGALURU_LNG_MIN = 77.3
BENGALURU_LNG_MAX = 77.9


def lat_lng_to_zone(lat: float, lng: float) -> str:
    """
    Convert a (lat, lng) coordinate to a zone_id string.
    Zone ID format: 'Z_{lat_idx}_{lng_idx}'
    """
    lat_idx = int((lat - BENGALURU_LAT_MIN) / LAT_STEP)
    lng_idx = int((lng - BENGALURU_LNG_MIN) / LNG_STEP)
    return f"Z_{lat_idx}_{lng_idx}"


def zone_to_center(zone_id: str) -> tuple:
    """
    Convert a zone_id back to the center (lat, lng) of that grid cell.
    """
    parts = zone_id.split("_")
    lat_idx = int(parts[1])
    lng_idx = int(parts[2])

    center_lat = BENGALURU_LAT_MIN + (lat_idx + 0.5) * LAT_STEP
    center_lng = BENGALURU_LNG_MIN + (lng_idx + 0.5) * LNG_STEP
    return (center_lat, center_lng)


def assign_zones(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign each row in the DataFrame to a zone based on its lat/lng.

    Args:
        df: DataFrame with 'latitude' and 'longitude' columns.

    Returns:
        DataFrame with added 'zone_id', 'zone_center_lat', 'zone_center_lng' columns.
    """
    print(f"[ZONE] Assigning {len(df):,} violations to {GRID_SIZE_METERS}m grid zones...")

    # Vectorized zone assignment
    lat_idx = ((df["latitude"].values - BENGALURU_LAT_MIN) / LAT_STEP).astype(int)
    lng_idx = ((df["longitude"].values - BENGALURU_LNG_MIN) / LNG_STEP).astype(int)

    df["zone_id"] = [f"Z_{la}_{lo}" for la, lo in zip(lat_idx, lng_idx)]

    # Calculate zone centers
    df["zone_center_lat"] = BENGALURU_LAT_MIN + (lat_idx + 0.5) * LAT_STEP
    df["zone_center_lng"] = BENGALURU_LNG_MIN + (lng_idx + 0.5) * LNG_STEP

    n_zones = df["zone_id"].nunique()
    print(f"[ZONE] ✅ Created {n_zones:,} unique zones")

    # Zone size distribution
    zone_sizes = df["zone_id"].value_counts()
    print(f"[ZONE] Violations per zone — min: {zone_sizes.min()}, median: {zone_sizes.median():.0f}, "
          f"max: {zone_sizes.max()}, mean: {zone_sizes.mean():.1f}")

    return df


def get_zone_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary table of all zones.

    Returns:
        DataFrame with columns: zone_id, center_lat, center_lng, violation_count
    """
    summary = df.groupby("zone_id").agg(
        center_lat=("zone_center_lat", "first"),
        center_lng=("zone_center_lng", "first"),
        violation_count=("id", "count"),
    ).reset_index()

    return summary.sort_values("violation_count", ascending=False)


if __name__ == "__main__":
    # Quick test
    test_coords = [
        (12.9344, 77.6117, "Forum Mall"),
        (12.9767, 77.5713, "Majestic"),
        (12.9784, 77.6408, "Indiranagar"),
    ]

    for lat, lng, name in test_coords:
        zone = lat_lng_to_zone(lat, lng)
        center = zone_to_center(zone)
        print(f"{name}: ({lat}, {lng}) → {zone} (center: {center[0]:.4f}, {center[1]:.4f})")
