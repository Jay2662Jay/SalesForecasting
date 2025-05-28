import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

# -------- CONFIG --------
FORECAST_DIR = "/Users/jaygamage/Downloads/r1.4/forecasts"
METRICS_PATH = "/Users/jaygamage/Downloads/r1.4/metrics.csv"
TRAIN_PATH = "/Users/jaygamage/Downloads/feature_outputs/train_featured.csv"

# -------- LOAD DATA --------
@st.cache_data
def load_data():
    metrics = pd.read_csv(METRICS_PATH)
    train = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    return metrics, train

metrics_df, train_df = load_data()
group_codes = sorted(metrics_df['group_code'].unique())

# -------- SIDEBAR --------
st.sidebar.title("📈 Tyre Sales Forecast Dashboard")
selected_group = st.sidebar.selectbox("Select Tyre Group", group_codes)

# -------- LOAD FORECAST DATA --------
forecast_file = os.path.join(FORECAST_DIR, f"{selected_group}_forecast.csv")
forecast_df = pd.read_csv(forecast_file, parse_dates=["date"])

# Filter actuals for the selected group
actual_df = train_df[train_df["group_code"] == selected_group][["date", "sales_quantity"]]

# -------- KPI METRICS --------
group_metrics = metrics_df[metrics_df['group_code'] == selected_group].iloc[0]
st.markdown(f"## 📊 Forecast Summary for Group {selected_group}")

col1, col2 = st.columns(2)
col1.metric("MAPE", f"{group_metrics['MAPE']:.2f}%")
col2.metric("R²", f"{group_metrics['R2']:.2f}")

# -------- FORECAST CHART --------
fig = go.Figure()

# Historical actuals
fig.add_trace(go.Scatter(
    x=actual_df['date'], 
    y=actual_df['sales_quantity'], 
    mode='lines+markers',
    name='Actual Sales',
    line=dict(color='blue')
))

# Forecasted sales
fig.add_trace(go.Scatter(
    x=forecast_df['date'], 
    y=forecast_df['predicted_sales_quantity'], 
    mode='lines+markers',
    name='Forecasted Sales',
    line=dict(color='orange', dash='dash')
))

fig.update_layout(
    title=f"Actual vs Forecasted Sales for Group {selected_group}",
    xaxis_title="Date",
    yaxis_title="Sales Quantity",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="simple_white"
)

st.plotly_chart(fig, use_container_width=True)

# -------- FORECAST TABLE --------
st.markdown("### 🔍 Forecast Table (2025)")
st.dataframe(forecast_df.style.format({"predicted_sales_quantity": "{:.0f}"}), use_container_width=True)

# -------- DOWNLOAD OPTION --------
csv = forecast_df.to_csv(index=False).encode()
st.download_button(
    label="📥 Download Forecast Data",
    data=csv,
    file_name=f"group_{selected_group}_forecast.csv",
    mime='text/csv'
)