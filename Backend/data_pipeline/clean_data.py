"""
GridLock AI — Data Cleaning & Preprocessing Pipeline
Cleans raw Bengaluru police parking violation data.
"""

import pandas as pd
import numpy as np
import ast
import re
import os


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV data."""
    print(f"[CLEAN] Loading raw data from: {filepath}")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"[CLEAN] Raw rows: {len(df):,}")
    return df


def clean_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with missing or invalid lat/lng coordinates."""
    initial = len(df)
    df = df.dropna(subset=["latitude", "longitude"])

    # Convert to numeric, coercing errors
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])

    # Bengaluru bounding box filter (approximate)
    # Lat: 12.7 - 13.2, Lng: 77.3 - 77.9
    df = df[
        (df["latitude"] >= 12.7) & (df["latitude"] <= 13.2) &
        (df["longitude"] >= 77.3) & (df["longitude"] <= 77.9)
    ]

    print(f"[CLEAN] Coordinate cleaning: {initial:,} → {len(df):,} rows (dropped {initial - len(df):,})")
    return df


def parse_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse created_datetime and extract temporal features."""
    df["created_datetime"] = pd.to_datetime(df["created_datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["created_datetime"])

    # Extract temporal features
    df["hour"] = df["created_datetime"].dt.hour
    df["day_of_week"] = df["created_datetime"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["month"] = df["created_datetime"].dt.month
    df["date"] = df["created_datetime"].dt.date
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Time period classification
    def classify_time_period(hour):
        if 6 <= hour < 10:
            return "morning_peak"
        elif 10 <= hour < 16:
            return "midday"
        elif 16 <= hour < 21:
            return "evening_peak"
        else:
            return "night"

    df["time_period"] = df["hour"].apply(classify_time_period)

    # Is peak hour? (morning 8-10, evening 5-8)
    df["is_peak_hour"] = ((df["hour"] >= 8) & (df["hour"] <= 10) |
                          (df["hour"] >= 17) & (df["hour"] <= 20)).astype(int)

    print(f"[CLEAN] Datetime features extracted. Columns added: hour, day_of_week, month, is_weekend, time_period, is_peak_hour")
    return df


def parse_violation_types(df: pd.DataFrame) -> pd.DataFrame:
    """Parse violation_type from JSON-like string to list, and create binary flags."""
    def safe_parse_violations(val):
        """Parse violation type strings like '["WRONG PARKING","NO PARKING"]'."""
        if pd.isna(val) or val == "NULL":
            return []
        try:
            # Try ast.literal_eval for proper Python list strings
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return [str(v).strip().upper() for v in parsed]
            return [str(parsed).strip().upper()]
        except (ValueError, SyntaxError):
            # Fallback: regex extraction
            matches = re.findall(r'"([^"]+)"', str(val))
            if matches:
                return [m.strip().upper() for m in matches]
            return [str(val).strip().upper()]

    df["violation_list"] = df["violation_type"].apply(safe_parse_violations)
    df["violation_count_per_record"] = df["violation_list"].apply(len)

    # Key violation type binary flags
    violation_flags = {
        "is_wrong_parking": "WRONG PARKING",
        "is_no_parking": "NO PARKING",
        "is_main_road_parking": "PARKING IN A MAIN ROAD",
        "is_road_crossing_parking": "PARKING NEAR ROAD CROSSING",
        "is_footpath_parking": "PARKING ON FOOTPATH",
        "is_double_parking": "DOUBLE PARKING",
    }

    for col_name, violation_str in violation_flags.items():
        df[col_name] = df["violation_list"].apply(
            lambda vlist: 1 if violation_str in vlist else 0
        )

    print(f"[CLEAN] Violation types parsed. Flags created: {list(violation_flags.keys())}")
    return df


def standardize_vehicle_type(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize vehicle_type to consistent categories."""
    vehicle_map = {
        "CAR": "CAR",
        "SCOOTER": "TWO_WHEELER",
        "MOTORCYCLE": "TWO_WHEELER",
        "BIKE": "TWO_WHEELER",
        "AUTO": "AUTO",
        "AUTO-RICKSHAW": "AUTO",
        "AUTORICKSHAW": "AUTO",
        "BUS": "BUS",
        "TRUCK": "HEAVY_VEHICLE",
        "LORRY": "HEAVY_VEHICLE",
        "TEMPO": "HEAVY_VEHICLE",
        "MAXI-CAB": "MAXI_CAB",
        "MAXICAB": "MAXI_CAB",
        "VAN": "VAN",
        "JEEP": "CAR",
        "SUV": "CAR",
        "TAXI": "TAXI",
        "CAB": "TAXI",
        "E-RICKSHAW": "AUTO",
    }

    df["vehicle_type_raw"] = df["vehicle_type"]
    df["vehicle_type_clean"] = df["vehicle_type"].str.strip().str.upper().map(vehicle_map)
    df["vehicle_type_clean"] = df["vehicle_type_clean"].fillna("OTHER")

    print(f"[CLEAN] Vehicle types standardized. Distribution:")
    print(df["vehicle_type_clean"].value_counts().to_string())
    return df


def parse_response_time(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate response time between violation creation and action taken."""
    df["action_taken_timestamp"] = pd.to_datetime(
        df["action_taken_timestamp"], errors="coerce", utc=True
    )

    # Response time in hours
    mask = df["action_taken_timestamp"].notna()
    df.loc[mask, "response_time_hours"] = (
        (df.loc[mask, "action_taken_timestamp"] - df.loc[mask, "created_datetime"])
        .dt.total_seconds() / 3600
    )

    # Cap at reasonable values (0 to 720 hours = 30 days)
    df["response_time_hours"] = df["response_time_hours"].clip(0, 720)

    valid = df["response_time_hours"].notna().sum()
    print(f"[CLEAN] Response time calculated for {valid:,} records. Median: {df['response_time_hours'].median():.1f} hours")
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate records."""
    initial = len(df)
    df = df.drop_duplicates(subset=["id"], keep="first")
    print(f"[CLEAN] Deduplication: {initial:,} → {len(df):,} rows (removed {initial - len(df):,} duplicates)")
    return df


def add_junction_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flag violations near junctions."""
    df["is_near_junction"] = (
        df["junction_name"].notna() &
        (df["junction_name"].str.strip().str.upper() != "NO JUNCTION")
    ).astype(int)

    junction_pct = df["is_near_junction"].mean() * 100
    print(f"[CLEAN] Junction flag: {junction_pct:.1f}% violations near a junction")
    return df


def clean_data(filepath: str) -> pd.DataFrame:
    """
    Run the full data cleaning pipeline.

    Args:
        filepath: Path to raw CSV file.

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    print("=" * 70)
    print("  GRIDLOCK AI — DATA CLEANING PIPELINE")
    print("=" * 70)

    df = load_raw_data(filepath)
    df = deduplicate(df)
    df = clean_coordinates(df)
    df = parse_datetime_features(df)
    df = parse_violation_types(df)
    df = standardize_vehicle_type(df)
    df = parse_response_time(df)
    df = add_junction_flag(df)

    # Final column selection (keep useful columns)
    keep_cols = [
        "id", "latitude", "longitude", "location",
        "vehicle_number", "vehicle_type_clean",
        "violation_list", "violation_count_per_record",
        "is_wrong_parking", "is_no_parking", "is_main_road_parking",
        "is_road_crossing_parking", "is_footpath_parking", "is_double_parking",
        "created_datetime", "hour", "day_of_week", "month", "date",
        "is_weekend", "time_period", "is_peak_hour",
        "police_station", "junction_name", "is_near_junction",
        "response_time_hours", "validation_status",
    ]

    # Only keep columns that exist
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    print(f"\n[CLEAN] ✅ Cleaning complete. Final shape: {df.shape}")
    print(f"[CLEAN] Columns: {list(df.columns)}")
    return df


if __name__ == "__main__":
    # Test run
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "Dataset",
        "jan to may police violation_anonymized791b166.csv"
    )
    df = clean_data(data_path)
    print(f"\nSample rows:")
    print(df.head(3).to_string())
