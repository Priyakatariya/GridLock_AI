import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import plotly.express as px
import os
import json

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="GridLock AI", layout="wide", page_icon="🚦")

# ==========================================
# LIGHT/DARK MODE TOGGLE & CSS
# ==========================================
theme_toggle = st.sidebar.toggle("🌙 Enable Dark Mode", value=True)

if theme_toggle:
    st.markdown("""
    <style>
        .stApp { background-color: #1A1F2B; color: #E0E6ED; }
        .stTabs [data-baseweb="tab-list"] { background-color: #242A38; border-radius: 8px; }
        div[data-testid="metric-container"] {
            background-color: #242A38;
            border: 1px solid #3A4150;
            padding: 15px;
            border-radius: 12px;
        }
        .status-box {
            background-color: #1B3B2B;
            color: #A3E4D7;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #2ECC71;
            margin-bottom: 25px;
        }
        .alert-banner {
            background: linear-gradient(90deg, #C0392B, #922B21);
            color: white;
            padding: 15px;
            border-radius: 8px;
            font-weight: bold;
            text-align: center;
            animation: pulse 2s infinite;
            margin-bottom: 20px;
        }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.8; } 100% { opacity: 1; } }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        .stApp { background-color: #F8F9FA; color: #2C3E50; }
        div[data-testid="metric-container"] {
            background-color: #FFFFFF;
            border: 1px solid #E5E7EB;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .status-box {
            background-color: #E8F8F5;
            color: #145A32;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #2ECC71;
            margin-bottom: 25px;
        }
        .alert-banner {
            background: linear-gradient(90deg, #E74C3C, #C0392B);
            color: white;
            padding: 15px;
            border-radius: 8px;
            font-weight: bold;
            text-align: center;
            animation: pulse 2s infinite;
            margin-bottom: 20px;
        }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.8; } 100% { opacity: 1; } }
    </style>
    """, unsafe_allow_html=True)

st.title("🚦 GridLock AI — Urban Congestion Command Center")

# ==========================================
# LOAD DATA (Optimized)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "..", "Backend", "outputs")

@st.cache_data
def load_data():
    raw_df = pd.read_csv(os.path.join(OUTPUTS_DIR, "cleaned_data_sample.csv"))
    raw_df = raw_df.dropna(subset=["latitude", "longitude"])
    
    impact_df = pd.read_csv(os.path.join(OUTPUTS_DIR, "zone_impact_scores.csv"))
    hotspot_df = pd.read_csv(os.path.join(OUTPUTS_DIR, "hotspot_summary.csv"))
    return raw_df, impact_df, hotspot_df

df, impact_df, hotspot_df = load_data()

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Map Simulation Controls")
show_heatmap = st.sidebar.checkbox("🔥 Show Heatmap Layer", True)
show_clusters = st.sidebar.checkbox("⭕ Show Hotspot Clusters", True)
heat_radius = st.sidebar.slider("Heatmap Intensity", 5, 25, 12)

st.sidebar.markdown("---")
st.sidebar.header("⏱️ Predictive AI Engine")
future_mins = st.sidebar.slider("Fast-Forward Time (Mins)", 0, 60, 0, step=15)

if future_mins > 0:
    alert_msg = f"⚠️ PREDICTIVE ALERT (T+{future_mins} mins): Spillover expected in High Risk Zones. Pre-emptive dispatch engaged."
    st.markdown(f"<div class='alert-banner'>{alert_msg}</div>", unsafe_allow_html=True)

# ==========================================
# HEADER UI
# ==========================================
st.markdown("##### Priority zones identified by the GridLock AI engine.")

total_zones = len(impact_df)
high_zones = len(impact_df[impact_df['severity'] == 'CRITICAL']) + len(impact_df[impact_df['severity'] == 'HIGH'])
med_zones = len(impact_df[impact_df['severity'] == 'MEDIUM'])

m1, m2, m3 = st.columns(3)
with m1: st.markdown(f"🔴 **High Risk Zones**\n# {high_zones}")
with m2: st.markdown(f"🟠 **Medium Risk Zones**\n# {med_zones}")
with m3: st.markdown(f"🟢 **Total Zones**\n# {total_zones}")

st.markdown("""
<div class="status-box">
    <h3>🟢 All Systems Operational</h3>
    <p>✅ <b>Spillover Prediction Engine Online</b></p>
    <p>✅ <b>Impact Score Engine Online</b></p>
    <p>✅ <b>Hotspot Detection Active</b></p>
    <p>✅ <b>Enforcement Prioritization Active</b></p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 3-TAB STRUCTURE
# ==========================================
tab1, tab2, tab3 = st.tabs(["🗺️ Live Congestion Map", "📊 ML Risk Analytics", "🚓 Priority Dispatch"])

# Dynamically escalate severity based on simulated impact_score
def update_severity(score):
    if score > 0.8: return "CRITICAL"
    if score > 0.6: return "HIGH"
    if score > 0.4: return "MEDIUM"
    return "LOW"
    
if future_mins > 0:
    simulated_df["severity"] = simulated_df["impact_score"].apply(update_severity)

# === TAB 1: OPTIMIZED FOLIUM MAP ===
with tab1:
    st.subheader("Live City Congestion Map")
    
    center_lat = simulated_df["center_lat"].mean()
    center_lon = simulated_df["center_lng"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    if show_heatmap:
        heat_sample = df.sample(min(15000, len(df))) if len(df) > 15000 else df
        heat_data = heat_sample[["latitude", "longitude"]].values.tolist()
        HeatMap(heat_data, radius=heat_radius, blur=heat_radius-2).add_to(m)

    if show_clusters:
        for _, row in simulated_df.iterrows():
            if row['severity'] in ["CRITICAL", "HIGH"]:
                folium.Circle(
                    location=[row["center_lat"], row["center_lng"]],
                    radius=min(600, 200 + (row['impact_score'] * 300)), 
                    color="red" if row['severity'] == "CRITICAL" else "orange",
                    fill=True,
                    fill_opacity=0.2,
                    tooltip=f"<b style='color:red;'>Zone {row['zone_id']}</b><br>Simulated Impact: {row['impact_score']:.2f}"
                ).add_to(m)

    sample_dots = df.sample(min(200, len(df)))
    for _, row in sample_dots.iterrows():
        violation = str(row.get("violation_list", ""))
        if "WRONG PARKING" in violation: color = "red"
        elif "NO PARKING" in violation: color = "orange"
        else: color = "green"

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.8,
            tooltip=f"Violation: {violation}"
        ).add_to(m)

    # Force re-render when controls change
    map_key = f"map_{heat_radius}_{future_mins}_{show_heatmap}_{show_clusters}"
    st_folium(m, width=1200, height=600, key=map_key)

# === TAB 2: ML RISK ANALYTICS ===
with tab2:
    st.subheader("🤖 Deep ML Risk Analysis & Top Predictions")
    
    col_c1, col_c2, col_c3 = st.columns(3)
    
    with col_c1:
        severity_counts = simulated_df['severity'].value_counts().reset_index()
        severity_counts.columns = ['severity', 'count']
        fig1 = px.pie(
            severity_counts, values='count', names='severity', hole=0.4,
            title="Severity Distribution",
            color='severity',
            color_discrete_map={"CRITICAL": "red", "HIGH": "orange", "MEDIUM": "yellow", "LOW": "green"},
            template="plotly_dark" if theme_toggle else "plotly_white"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_c2:
        top_10 = simulated_df.sort_values(by="impact_score", ascending=False).head(10)
        top_10["zone_id_str"] = "Zone " + top_10["zone_id"].astype(str)
        fig2 = px.bar(
            top_10, x="zone_id_str", y="impact_score", color="severity",
            title="Top 10 Critical Zones",
            color_discrete_map={"CRITICAL": "red", "HIGH": "orange", "MEDIUM": "yellow", "LOW": "green"},
            template="plotly_dark" if theme_toggle else "plotly_white"
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_c3:
        fig3 = px.scatter(
            simulated_df, x="violation_count", y="impact_score", 
            color="severity", size="risk_score", hover_data=["zone_id"],
            title="Impact vs Violations",
            color_discrete_map={"CRITICAL": "red", "HIGH": "orange", "MEDIUM": "yellow", "LOW": "green"},
            template="plotly_dark" if theme_toggle else "plotly_white"
        )
        st.plotly_chart(fig3, use_container_width=True)

# === TAB 3: DISPATCH CONSOLE ===
with tab3:
    st.subheader("🚓 Automated Action Output")
    
    def format_urgency(val):
        if val == "CRITICAL": return "🚨 Dispatch Tow Truck ASAP"
        if val == "HIGH": return "🚓 Send Patrol Unit"
        return "🟢 Issue E-Challan"

    dispatch_df = simulated_df.copy()
    dispatch_df["Recommended_Action"] = dispatch_df["severity"].apply(format_urgency)
    
    st.markdown("##### Real-Time Dispatch Requirements")
    action_summary = dispatch_df['Recommended_Action'].value_counts().reset_index()
    action_summary.columns = ['Action', 'Count']
    
    fig_dispatch = px.bar(
        action_summary, x="Count", y="Action", orientation='h',
        title="Units Needed For Immediate Dispatch",
        color="Action",
        color_discrete_map={"🚨 Dispatch Tow Truck ASAP": "red", "🚓 Send Patrol Unit": "orange", "🟢 Issue E-Challan": "green"},
        template="plotly_dark" if theme_toggle else "plotly_white"
    )
    st.plotly_chart(fig_dispatch, use_container_width=True)
    
    st.markdown("##### Priority Action Table")
    
    c1, c2 = st.columns(2)
    with c1:
        severity_filter = st.multiselect("Filter by Severity:", ["CRITICAL", "HIGH", "MEDIUM", "LOW"], default=["CRITICAL", "HIGH", "MEDIUM", "LOW"])
    with c2:
        sort_by = st.selectbox("Sort Priority By:", ["Highest Impact", "Most Violations"])
        
    if severity_filter:
        dispatch_df = dispatch_df[dispatch_df["severity"].isin(severity_filter)]
        
    if sort_by == "Highest Impact":
        dispatch_df = dispatch_df.sort_values(by="impact_score", ascending=False)
    else:
        dispatch_df = dispatch_df.sort_values(by="violation_count", ascending=False)
    
    display_cols = ["zone_id", "severity", "risk_score", "impact_score", "Recommended_Action"]
    
    dispatch_df["risk_score"] = dispatch_df["risk_score"].round(4)
    dispatch_df["impact_score"] = dispatch_df["impact_score"].round(4)
    
    def highlight_critical(row):
        if row.severity == 'CRITICAL':
            bg_color = "rgba(255, 0, 0, 0.2)" if theme_toggle else "rgba(255, 0, 0, 0.1)"
            return [f'background-color: {bg_color}'] * len(row)
        return [''] * len(row)
        
    st.dataframe(
        dispatch_df[display_cols].style.apply(highlight_critical, axis=1),
        use_container_width=True,
        hide_index=True,
        height=500
    )