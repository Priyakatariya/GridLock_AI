# 🚦 GridLock AI

**Urban Congestion Command Center & AI-Driven Parking Intelligence**

GridLock AI is an intelligent dashboard designed to tackle the problem of poor visibility on parking-induced congestion. It detects illegal parking hotspots, quantifies their impact on traffic flow using Machine Learning, and provides actionable insights for targeted traffic police enforcement.

## 🌟 Key Features

1. **🗺️ Live Congestion Map & Hotspots**
   - Visualizes live traffic congestion and illegal parking density using interactive custom canvas maps.
   - Dynamically highlights high-priority choke points.

2. **🤖 Spillover Risk Engine (ML Forecasting)**
   - Uses advanced data pipelines to automatically detect critical "hotspots" from raw lat/long data.
   - Calculates a "Congestion Impact Score" and forecasts traffic conditions 24 hours ahead based on historical patterns.

3. **🚨 Enforcement Priority Dispatch**
   - Automatically ranks illegally parked vehicles and zones by severity (e.g., CRITICAL vs LOW).
   - Suggests immediate actionable insights: **Dispatch Tow Truck** vs. **Send Patrol**.

---

## 📂 Project Structure

```text
GridLock_AI/
├── Backend/                 # Machine Learning models & data pipelines
│   ├── models/              # Traffic forecasting & clustering scripts
│   └── outputs/             # Processed datasets ready for the dashboard
├── static/                  # Web dashboard assets (HTML, CSS, JS)
│   ├── index.html           # Main UI structure
│   ├── css/style.css        # Premium dark-theme UI styles
│   └── js/                  # App logic & Chart.js data visualization
├── server.py                # Flask API server providing data to the frontend
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

*(Note: The `Frontend/app.py` Streamlit file remains in the repository as a legacy reference, but the primary Hackathon submission runs natively via the Flask server for a more premium UI).*

---

## 🚀 How to Run the Command Center

1. **Clone the repository:**
   ```bash
   git clone https://github.com/poornima200631/GridLock_AI.git
   cd GridLock_AI
   ```

2. **Install Dependencies:**
   Make sure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: The new UI relies on `Flask` and `flask-cors`. These have been added to the requirements, or you can install them directly via `pip install flask flask-cors`)*

3. **Start the API & Web Server:**
   From the root directory, start the Python server:
   ```bash
   python server.py
   ```

4. **View in Browser:**
   Open your web browser and navigate to `http://localhost:5000` to view the live dashboard.

---

## 👥 Hackathon Team Roles
- **Person 1 (ML Engine):** Data Processing, Feature Extraction & Clustering Logic
- **Person 2 (Geo Engine):** Geospatial Intelligence & Traffic Modeling
- **Person 3 (Dashboard):** Custom Web UI Development, API Integration & Final Product Flow
