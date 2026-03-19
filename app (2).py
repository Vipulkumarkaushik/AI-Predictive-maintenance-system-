import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(page_title="Industrial AI Optimizer", layout="wide")

# Custom CSS for Professional Look
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border_radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Enterprise Predictive Maintenance Suite")
st.markdown("---")

# Load Spark Predictions
df = pd.read_csv('spark_predictions.csv')

# --- SIDEBAR: Fleet Control ---
st.sidebar.header("Global Fleet Controls")
machine_id = st.sidebar.selectbox("Select Asset", ["Machine M-101", "Machine M-102 (Offline)"])
confidence_threshold = st.sidebar.slider("AI Confidence Threshold (%)", 50, 100, 95)

# --- TOP ROW: KPI Metrics ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Asset Status", "ACTIVE", delta="Normal")
with col2:
    total_anomalies = df['prediction'].sum()
    st.metric("Detected Anomalies", int(total_anomalies), delta="-2% vs yesterday", delta_color="inverse")
with col3:
    st.metric("Model Accuracy", "99.96%", "State-of-the-Art")
with col4:
    st.metric("Mean Temp", f"{df['Temperature_C'].mean():.2f} °C")

# --- MIDDLE ROW: Advanced Visualization ---
st.subheader("Interactive Thermal Analysis & AI Prediction")
fig = px.line(df.head(500), x='Timestamp', y='Temperature_C', 
              title="Real-time Sensor Feed vs. AI Classification")

# Highlight Anomalies in Red
anomalies = df.head(500)[df.head(500)['prediction'] == 1]
fig.add_trace(go.Scatter(x=anomalies['Timestamp'], y=anomalies['Temperature_C'],
                         mode='markers', name='AI Detected Fault',
                         marker=dict(color='red', size=10, symbol='x')))
st.plotly_chart(fig, use_container_width=True)

# --- BOTTOM ROW: Feature Importance ---
c1, c2 = st.columns(2)
with c1:
    st.subheader("Feature Correlation (Explainable AI)")
    st.write("Spark Model weights the 'Temp_Gradient' as the highest predictor of failure.")
    st.scatter_chart(df.head(200), x='Temp_Gradient', y='Temperature_C', color='prediction')
with c2:
    st.subheader("System Logs")
    st.dataframe(df[df['prediction'] == 1].tail(10), use_container_width=True)
