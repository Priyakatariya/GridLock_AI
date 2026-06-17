"""
GridLock AI — Bengaluru Points of Interest (POI) Database
Real-world coordinates for metro stations, malls, hospitals, markets, bus stands, schools.
Used for proximity-based feature engineering.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Each POI: (name, latitude, longitude, category)
# ─────────────────────────────────────────────────────────────────────────────

BENGALURU_POIS = [
    # ── Metro Stations (Purple Line) ──────────────────────────────────────
    ("Majestic Metro", 12.9767, 77.5713, "metro_station"),
    ("MG Road Metro", 12.9756, 77.6065, "metro_station"),
    ("Indiranagar Metro", 12.9784, 77.6408, "metro_station"),
    ("Trinity Metro", 12.9727, 77.6197, "metro_station"),
    ("Cubbon Park Metro", 12.9763, 77.5929, "metro_station"),
    ("Vidhana Soudha Metro", 12.9794, 77.5913, "metro_station"),
    ("Baiyappanahalli Metro", 12.9898, 77.6530, "metro_station"),
    ("Mysore Road Metro", 12.9592, 77.5397, "metro_station"),
    ("Peenya Metro", 13.0313, 77.5194, "metro_station"),
    ("Yeshwanthpur Metro", 13.0218, 77.5470, "metro_station"),

    # ── Metro Stations (Green Line) ──────────────────────────────────────
    ("Yelachenahalli Metro", 12.9004, 77.5701, "metro_station"),
    ("Jayanagar Metro", 12.9289, 77.5828, "metro_station"),
    ("South End Circle Metro", 12.9424, 77.5762, "metro_station"),
    ("Rajajinagar Metro", 12.9918, 77.5545, "metro_station"),
    ("Nagasandra Metro", 13.0440, 77.5152, "metro_station"),
    ("Mantri Square Metro", 12.9851, 77.5706, "metro_station"),

    # ── Malls / Commercial Hubs ──────────────────────────────────────────
    ("Forum Mall Koramangala", 12.9344, 77.6117, "mall"),
    ("Phoenix Marketcity", 12.9976, 77.6969, "mall"),
    ("Mantri Square Mall", 12.9858, 77.5707, "mall"),
    ("Orion Mall", 12.9921, 77.5565, "mall"),
    ("VR Bengaluru", 12.9955, 77.6964, "mall"),
    ("UB City", 12.9716, 77.5964, "mall"),
    ("Garuda Mall", 12.9703, 77.6099, "mall"),
    ("Lulu Mall", 13.0140, 77.5510, "mall"),
    ("GT World Mall", 12.9586, 77.6478, "mall"),
    ("Elements Mall", 12.9095, 77.5858, "mall"),

    # ── Hospitals ─────────────────────────────────────────────────────────
    ("Manipal Hospital", 12.9586, 77.6484, "hospital"),
    ("St. Johns Hospital", 12.9296, 77.6189, "hospital"),
    ("Fortis Hospital", 12.9065, 77.5987, "hospital"),
    ("Apollo Hospital", 12.9508, 77.5991, "hospital"),
    ("Narayana Health", 12.8794, 77.5994, "hospital"),
    ("Columbia Asia Hebbal", 13.0350, 77.5835, "hospital"),
    ("Sakra World Hospital", 12.9322, 77.6895, "hospital"),
    ("Nimhans", 12.9427, 77.5960, "hospital"),

    # ── Markets / Commercial Streets ─────────────────────────────────────
    ("KR Market", 12.9634, 77.5770, "market"),
    ("Commercial Street", 12.9827, 77.6078, "market"),
    ("Chickpet Market", 12.9660, 77.5762, "market"),
    ("Jayanagar 4th Block Market", 12.9250, 77.5830, "market"),
    ("Malleswaram Market", 13.0035, 77.5680, "market"),
    ("Koramangala Market", 12.9347, 77.6235, "market"),
    ("HSR Layout Market", 12.9116, 77.6389, "market"),
    ("BTM Layout Market", 12.9134, 77.6101, "market"),

    # ── Bus Stands ────────────────────────────────────────────────────────
    ("Majestic Bus Stand", 12.9770, 77.5720, "bus_stand"),
    ("Shantinagar Bus Stand", 12.9561, 77.5984, "bus_stand"),
    ("KBS (Kempegowda Bus Station)", 12.9771, 77.5724, "bus_stand"),
    ("Banashankari Bus Stand", 12.9152, 77.5730, "bus_stand"),
    ("Whitefield Bus Depot", 12.9698, 77.7500, "bus_stand"),
    ("Silk Board Junction", 12.9173, 77.6228, "bus_stand"),

    # ── Schools / Colleges ────────────────────────────────────────────────
    ("IISc Bangalore", 13.0219, 77.5671, "school"),
    ("Christ University", 12.9348, 77.6068, "school"),
    ("St. Joseph's College", 12.9502, 77.6008, "school"),
    ("RV College", 12.9236, 77.4989, "school"),
    ("PES University", 12.9341, 77.5352, "school"),
    ("BMS College", 12.9416, 77.5651, "school"),

    # ── Event / Convention Venues ─────────────────────────────────────────
    ("Palace Grounds", 13.0000, 77.5750, "event_venue"),
    ("Kanteerava Stadium", 12.9617, 77.5951, "event_venue"),
    ("Chinnaswamy Stadium", 12.9790, 77.5999, "event_venue"),
    ("KTPO Convention Center", 12.9568, 77.7395, "event_venue"),
    ("Bangalore International Exhibition Centre", 12.9720, 77.6938, "event_venue"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Quick-access category lookup
# ─────────────────────────────────────────────────────────────────────────────

def get_pois_by_category(category: str) -> list:
    """Return POIs filtered by category."""
    return [(name, lat, lng) for name, lat, lng, cat in BENGALURU_POIS if cat == category]


def get_all_poi_coords() -> list:
    """Return list of (lat, lng, category) for all POIs."""
    return [(lat, lng, cat) for _, lat, lng, cat in BENGALURU_POIS]


def get_poi_categories() -> list:
    """Return all unique POI categories."""
    return list(set(cat for _, _, _, cat in BENGALURU_POIS))


if __name__ == "__main__":
    print(f"Total POIs: {len(BENGALURU_POIS)}")
    for cat in sorted(get_poi_categories()):
        count = len(get_pois_by_category(cat))
        print(f"  {cat}: {count}")
