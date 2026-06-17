"""
GridLock AI — Flask REST API
Serves zone risk scores, impact scores, and hotspot data.
"""

import os
import sys
import json
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

# Add Backend to path
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────────────────────────────────────
# Load pre-computed data
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(BACKEND_DIR, "outputs")

def load_data():
    """Load pre-computed zone data from pipeline outputs."""
    full_data_path = os.path.join(OUTPUT_DIR, "zone_full_data.csv")
    metrics_path = os.path.join(OUTPUT_DIR, "model_metrics.json")
    summary_path = os.path.join(OUTPUT_DIR, "pipeline_summary.json")

    if not os.path.exists(full_data_path):
        print(f"[API] ERROR: No data found at {full_data_path}")
        print("[API] Run `python run_pipeline.py` first to generate data.")
        return None, None, None

    zone_data = pd.read_csv(full_data_path)

    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)

    print(f"[API] Loaded {len(zone_data):,} zones from {full_data_path}")
    return zone_data, metrics, summary


# Global data store
ZONE_DATA, MODEL_METRICS, PIPELINE_SUMMARY = load_data()


# ─────────────────────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "zones_loaded": len(ZONE_DATA) if ZONE_DATA is not None else 0,
    })


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Return overall pipeline statistics."""
    if ZONE_DATA is None:
        return jsonify({"error": "No data loaded. Run pipeline first."}), 503

    stats = {
        "total_zones": len(ZONE_DATA),
        "severity_distribution": ZONE_DATA["severity"].value_counts().to_dict(),
        "avg_risk_score": float(ZONE_DATA["risk_score"].mean()),
        "avg_impact_score": float(ZONE_DATA["impact_score"].mean()),
        "total_violations": int(ZONE_DATA["violation_count"].sum()),
        "model_metrics": MODEL_METRICS,
        "pipeline_summary": PIPELINE_SUMMARY,
    }
    return jsonify(stats)


@app.route("/api/zones", methods=["GET"])
def get_zones():
    """
    Return all zones with risk & impact scores.
    Query params:
        severity: filter by severity (CRITICAL, HIGH, MEDIUM, LOW)
        min_risk: minimum risk_score threshold
        limit: max number of results (default: all)
    """
    if ZONE_DATA is None:
        return jsonify({"error": "No data loaded. Run pipeline first."}), 503

    df = ZONE_DATA.copy()

    # Filters
    severity = request.args.get("severity")
    if severity:
        severity_list = [s.strip().upper() for s in severity.split(",")]
        df = df[df["severity"].isin(severity_list)]

    min_risk = request.args.get("min_risk", type=float)
    if min_risk is not None:
        df = df[df["risk_score"] >= min_risk]

    min_impact = request.args.get("min_impact", type=float)
    if min_impact is not None:
        df = df[df["impact_score"] >= min_impact]

    # Sort by enforcement priority
    df = df.sort_values("enforcement_priority")

    # Limit
    limit = request.args.get("limit", type=int)
    if limit:
        df = df.head(limit)

    # Select output columns
    output_cols = [
        "zone_id", "center_lat", "center_lng", "violation_count",
        "risk_score", "impact_score", "severity", "enforcement_priority",
    ]
    output_cols = [c for c in output_cols if c in df.columns]

    records = df[output_cols].to_dict(orient="records")

    return jsonify({
        "count": len(records),
        "zones": records,
    })


@app.route("/api/hotspots", methods=["GET"])
def get_hotspots():
    """
    Return top-N critical enforcement zones.
    Query params:
        n: number of hotspots to return (default: 20)
    """
    if ZONE_DATA is None:
        return jsonify({"error": "No data loaded. Run pipeline first."}), 503

    n = request.args.get("n", default=20, type=int)

    hotspots = ZONE_DATA.nsmallest(n, "enforcement_priority")

    output_cols = [
        "zone_id", "center_lat", "center_lng", "violation_count",
        "risk_score", "impact_score", "severity", "enforcement_priority",
        "wrong_parking_ratio", "main_road_parking_ratio",
        "peak_hour_ratio", "repeat_offender_ratio",
        "nearest_metro_dist", "nearest_mall_dist", "poi_count_500m",
    ]
    output_cols = [c for c in output_cols if c in hotspots.columns]

    records = hotspots[output_cols].to_dict(orient="records")

    return jsonify({
        "count": len(records),
        "hotspots": records,
    })


@app.route("/api/zone/<zone_id>", methods=["GET"])
def get_zone_detail(zone_id):
    """Return detailed information for a specific zone."""
    if ZONE_DATA is None:
        return jsonify({"error": "No data loaded. Run pipeline first."}), 503

    zone = ZONE_DATA[ZONE_DATA["zone_id"] == zone_id]

    if zone.empty:
        return jsonify({"error": f"Zone '{zone_id}' not found"}), 404

    # Convert to dict, handling NaN values
    zone_dict = zone.iloc[0].to_dict()
    zone_dict = {
        k: (None if pd.isna(v) else v)
        for k, v in zone_dict.items()
    }

    return jsonify({"zone": zone_dict})


@app.route("/api/heatmap", methods=["GET"])
def get_heatmap_data():
    """
    Return data formatted for heatmap visualization.
    Returns: list of {lat, lng, weight} where weight is impact_score.
    """
    if ZONE_DATA is None:
        return jsonify({"error": "No data loaded. Run pipeline first."}), 503

    min_impact = request.args.get("min_impact", default=0.0, type=float)
    df = ZONE_DATA[ZONE_DATA["impact_score"] >= min_impact]

    heatmap = [
        {
            "lat": float(row["center_lat"]),
            "lng": float(row["center_lng"]),
            "weight": float(row["impact_score"]),
            "severity": row["severity"],
            "violations": int(row["violation_count"]),
            "zone_id": row["zone_id"],
        }
        for _, row in df.iterrows()
    ]

    return jsonify({
        "count": len(heatmap),
        "heatmap": heatmap,
    })


@app.route("/api/search", methods=["GET"])
def search_nearby():
    """
    Find zones near a given coordinate.
    Query params:
        lat: latitude
        lng: longitude
        radius: search radius in km (default: 1)
    """
    if ZONE_DATA is None:
        return jsonify({"error": "No data loaded. Run pipeline first."}), 503

    lat = request.args.get("lat", type=float)
    lng = request.args.get("lng", type=float)
    radius_km = request.args.get("radius", default=1.0, type=float)

    if lat is None or lng is None:
        return jsonify({"error": "lat and lng are required query parameters."}), 400

    import numpy as np

    # Approximate distance using Haversine
    R = 6371  # Earth's radius in km
    dlat = np.radians(ZONE_DATA["center_lat"] - lat)
    dlng = np.radians(ZONE_DATA["center_lng"] - lng)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat)) * np.cos(np.radians(ZONE_DATA["center_lat"])) *
         np.sin(dlng / 2) ** 2)
    dist_km = 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    mask = dist_km <= radius_km
    nearby = ZONE_DATA[mask].copy()
    nearby["distance_km"] = dist_km[mask].round(3)
    nearby = nearby.sort_values("distance_km")

    output_cols = [
        "zone_id", "center_lat", "center_lng", "distance_km",
        "violation_count", "risk_score", "impact_score", "severity",
    ]
    output_cols = [c for c in output_cols if c in nearby.columns]

    records = nearby[output_cols].to_dict(orient="records")

    return jsonify({
        "query": {"lat": lat, "lng": lng, "radius_km": radius_km},
        "count": len(records),
        "zones": records,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[API] Starting GridLock AI API server...")
    print("[API] Endpoints:")
    print("  GET /api/health        — Health check")
    print("  GET /api/stats         — Overall statistics")
    print("  GET /api/zones         — All zones (with filters)")
    print("  GET /api/hotspots      — Top-N hotspots")
    print("  GET /api/zone/<id>     — Zone detail")
    print("  GET /api/heatmap       — Heatmap data")
    print("  GET /api/search        — Search by coordinates")
    print()

    app.run(host="0.0.0.0", port=5000, debug=True)
