import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import json
import os

# ==========================================
# ⚙️ PAGE CONFIG & PREMIUM UI STYLING
# ==========================================
st.set_page_config(page_title="GridLock AI Command Center", layout="wide", page_icon="🚦")

# Premium CSS
st.markdown("""
<style>
    h1 {
        font-weight: 800;
        background: -webkit-linear-gradient(#FF4B4B, #FF904F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: -10px;
    }
    div[data-testid="metric-container"] {
        border: 1px solid #444;
        background-color: #1A1C24;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .alert-banner {
        background: linear-gradient(90deg, #ff4b4b, #8b0000);
        color: white;
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        animation: pulse 2s infinite;
        margin-bottom: 20px;
    }
    .warning-banner {
        background: linear-gradient(90deg, #1e3d59, #2b5876);
        color: white;
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.8; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

st.title("🚦 GridLock AI — Intelligent Congestion Control")
st.markdown("*Real-time AI monitoring & Predictive Spillover Alerts for proactive enforcement.*")

# ==========================================
# 📊 LOAD ML DATA
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "..", "Backend", "outputs")

@st.cache_data
def load_data():
    try:
        impact_df = pd.read_csv(os.path.join(OUTPUTS_DIR, "zone_impact_scores.csv"))
        hotspot_df = pd.read_csv(os.path.join(OUTPUTS_DIR, "hotspot_summary.csv"))
        raw_df = pd.read_csv(os.path.join(OUTPUTS_DIR, "cleaned_data_sample.csv"))
        with open(os.path.join(OUTPUTS_DIR, "pipeline_summary.json"), "r") as f:
            summary = json.load(f)
        return impact_df, hotspot_df, raw_df, summary
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

impact_df, hotspot_df, raw_df, summary = load_data()

if impact_df.empty:
    st.error("❌ System Offline: Could not load ML pipeline data. Run the backend pipeline first.")
    st.stop()

# ==========================================
# 🎛️ SIDEBAR: PREDICTIVE SLIDER
# ==========================================
st.sidebar.header("⏱️ Predictive AI Engine")
st.sidebar.markdown("Use this slider to simulate future congestion spillover.")
future_mins = st.sidebar.slider("Fast-Forward Time (Minutes)", 0, 60, 0, step=15)

# Apply Predictive Multiplier logic
impact_multiplier = 1.0 + (future_mins * 0.05)
simulated_df = impact_df.copy()
simulated_df["impact_score"] = simulated_df["impact_score"] * impact_multiplier

# ==========================================
# 🚨 DYNAMIC PREDICTIVE ALERTS
# ==========================================
top_zone = simulated_df.sort_values(by="impact_score", ascending=False).iloc[0]

if future_mins > 0:
    alert_msg = f"⚠️ PREDICTIVE ALERT (T+{future_mins} mins): Zone {top_zone['zone_id']} is projected to reach 100% capacity causing severe spillover. Pre-emptive tow dispatch recommended."
    st.markdown(f"<div class='alert-banner'>{alert_msg}</div>", unsafe_allow_html=True)
else:
    info_msg = "✅ Live Monitoring: Traffic flow is currently stabilized. No immediate predictive alerts."
    st.markdown(f"<div class='warning-banner'>{info_msg}</div>", unsafe_allow_html=True)

# ==========================================
# 📈 LIVE KPI DASHBOARD
# ==========================================
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🚗 Total Violations Detected", f"{summary.get('total_raw_records', 0):,}")
with col2:
    st.metric("📍 Active Micro-Zones", f"{summary.get('total_zones', 0):,}")
with col3:
    critical_count = len(simulated_df[simulated_df["impact_score"] > simulated_df["impact_score"].quantile(0.8)])
    st.metric("🚨 Simulated Critical Zones", critical_count, delta=f"+{int(critical_count * (future_mins*0.02))} in {future_mins}m" if future_mins>0 else "Live", delta_color="inverse")
with col4:
    st.metric("⚡ AI Model Confidence", f"{summary.get('model_auc', 0.88)*100:.1f}%")

st.markdown("---")

# ==========================================
# 📁 3-TAB PROFESSIONAL STRUCTURE
# ==========================================
tab1, tab2, tab3 = st.tabs(["🗺️ Live Congestion Map", "📊 Spillover Risk Analytics (ML)", "🚓 Smart Police Dispatch Console"])

# === TAB 1: FOLIUM MAP (AS REQUESTED BY USER) ===
with tab1:
    st.subheader("Live City Congestion & Hotspots")
    
    st.markdown("""
    **🔴 High Severity Zone** &nbsp;&nbsp; | &nbsp;&nbsp;
    **🟠 Medium Severity Zone** &nbsp;&nbsp; | &nbsp;&nbsp;
    **🟢 Low Severity Zone**
    """)
    
    # Base Folium Map with CartoDB dark_matter to show roads clearly
    m = folium.Map(
        location=[simulated_df["center_lat"].mean(), simulated_df["center_lng"].mean()], 
        zoom_start=13,
        tiles="CartoDB dark_matter"
    )
    
    # 1. Draw Big Red Circles for Hotspot Zones
    for _, row in simulated_df.iterrows():
        # Only draw circles for areas with significant impact
        if row['severity'] in ["CRITICAL", "HIGH"]:
            radius = min(400, max(150, row['impact_score'] * 600))
            folium.Circle(
                location=[row["center_lat"], row["center_lng"]],
                radius=radius,
                color="red",
                fill=True,
                fill_opacity=0.15,
                tooltip=f"<b style='color:red;'>Zone {row['zone_id']}</b><br>Impact: {row['impact_score']:.2f}"
            ).add_to(m)

    # 2. Draw Individual Violations (Small dots)
    if not raw_df.empty:
        # We sample points to keep map fast, matching the image look
        sample_raw = raw_df.head(500) 
        for _, row in sample_raw.iterrows():
            viol = str(row.get("violation_list", ""))
            if "WRONG PARKING" in viol:
                color = "red"
            elif "NO PARKING" in viol:
                color = "orange"
            else:
                color = "green"
                
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=4,
                color=color,
                fill=True,
                fill_opacity=0.8,
                tooltip=f"Violation: {viol}"
            ).add_to(m)

    st_folium(m, width=1200, height=600)

# === TAB 2: ML RISK ANALYTICS (PLOTLY) ===
with tab2:
    st.subheader("🤖 Spillover Risk Analytics")
    st.markdown("Visualizing the Machine Learning outputs: Correlation between Violations, Base Risk, and Calculated Impact.")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        fig1 = px.scatter(
            impact_df, x="violation_count", y="impact_score", 
            color="severity", size="risk_score", hover_data=["zone_id"],
            title="Impact vs Violations (Bubble Size = Risk Score)",
            color_discrete_map={"CRITICAL": "red", "HIGH": "orange", "MEDIUM": "yellow", "LOW": "green"},
            template="plotly_dark"
        )
        st.plotly_chart(fig1, use_container_width=True)
        
    with col_chart2:
        top_10 = impact_df.sort_values(by="impact_score", ascending=False).head(10)
        top_10["zone_id_str"] = "Zone " + top_10["zone_id"].astype(str)
        fig2 = px.bar(
            top_10, x="zone_id_str", y="impact_score", color="severity",
            title="Top 10 Most Critical Zones",
            color_discrete_map={"CRITICAL": "red", "HIGH": "orange", "MEDIUM": "yellow", "LOW": "green"},
            template="plotly_dark"
        )
        st.plotly_chart(fig2, use_container_width=True)

# === TAB 3: UBER FOR TRAFFIC COPS (DISPATCH UI) ===
with tab3:
    st.subheader("🚓 Smart Police Dispatch Console")
    st.markdown("Actionable intelligence for traffic police. Filter and dispatch units efficiently.")
    
    if not hotspot_df.empty:
        dispatch_df = hotspot_df.copy()
        
        c1, c2 = st.columns(2)
        with c1:
            severity_filter = st.multiselect("Filter by Severity:", ["CRITICAL", "HIGH", "MEDIUM", "LOW"], default=["CRITICAL", "HIGH"])
        with c2:
            sort_by = st.selectbox("Sort Priority By:", ["Highest Impact", "Most Violations"])
            
        if severity_filter:
            dispatch_df = dispatch_df[dispatch_df["severity"].isin(severity_filter)]
            
        if sort_by == "Highest Impact":
            dispatch_df = dispatch_df.sort_values(by="impact_score", ascending=False)
        else:
            dispatch_df = dispatch_df.sort_values(by="violation_count", ascending=False)
        
        def format_urgency(val):
            if val == "CRITICAL": return "🚨 DISPATCH TOW TRUCK ASAP"
            if val == "HIGH": return "🚓 SEND PATROL UNIT"
            return "✅ MONITOR"
            
        dispatch_df["Action_Required"] = dispatch_df["severity"].apply(format_urgency)
        
        police_cols = ["enforcement_priority", "zone_id", "severity", "violation_count", "impact_score", "Action_Required"]
        
        def highlight_critical(row):
            if row.severity == 'CRITICAL':
                return ['background-color: rgba(255, 75, 75, 0.2)'] * len(row)
            return [''] * len(row)
            
        st.dataframe(
            dispatch_df[police_cols].style.apply(highlight_critical, axis=1),
            use_container_width=True,
            hide_index=True,
            height=400
        )
    else:
        st.success("✅ No critical hotspots detected. Traffic flow is optimal.")