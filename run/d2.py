import streamlit as st
import pandas as pd
import plotly.express as px
import os
from glob import glob

# -------- CONFIG --------
FORECAST_DIR = "/Users/jaygamage/Downloads/r1.4/forecasts"
METRICS_PATH = "/Users/jaygamage/Downloads/r1.4/metrics.csv"
TRAIN_PATH = "/Users/jaygamage/Downloads/feature_outputs/train_featured.csv"

# -------- STREAMLIT THEME --------
st.set_page_config(page_title="Tyre Sales Forecast", layout="wide")

# -------- LOAD DATA --------
@st.cache_data
def load_data():
    metrics = pd.read_csv(METRICS_PATH)
    train = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    forecast_files = glob(os.path.join(FORECAST_DIR, "*_forecast.csv"))
    forecast_list = [pd.read_csv(f, parse_dates=["date"]) for f in forecast_files]
    forecast_df = pd.concat(forecast_list, ignore_index=True)
    return metrics, train, forecast_df

metrics_df, train_df, forecast_df = load_data()

# -------- SIDEBAR NAV --------
st.sidebar.title("🛞 CKT Tyre Sales Dashboard")
menu = st.sidebar.radio("Navigation", ["📊 Dashboard", "📈 Cat-wise Forecast", "📂 Raw Data View"])

# -------- PAGE: DASHBOARD --------
if menu == "📊 Dashboard":
    st.title("📊 Sales Forecast Overview")

    # ---- Monthly Sales Trend ----
    monthly_sales = train_df.copy()
    monthly_sales["month"] = monthly_sales["date"].dt.month
    monthly_summary = monthly_sales.groupby("month")["sales_quantity"].sum().reset_index()

    st.markdown("### 📆 Sales Order by Month")
    fig1 = px.line(monthly_summary, x="month", y="sales_quantity", markers=True,
                   labels={"month": "Month", "sales_quantity": "Sales Quantity"},
                   color_discrete_sequence=["#F97316"])
    fig1.update_layout(template="plotly_dark")
    st.plotly_chart(fig1, use_container_width=True)

    # ---- Top Selling Groups ----
    top_sellers = train_df.groupby("group_code")["sales_quantity"].sum().reset_index()
    top_sellers = top_sellers.sort_values("sales_quantity", ascending=False).head(7)

    # ---- Forecasted Group Summary ----
    forecast_group = forecast_df.groupby("group_code")["predicted_sales_quantity"].sum().reset_index()
    forecast_group = forecast_group.sort_values("predicted_sales_quantity", ascending=False).head(5)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🥇 Top Selling Tyre Groups")
        fig2 = px.bar(top_sellers, x="sales_quantity", y="group_code", orientation="h",
                      labels={"sales_quantity": "Quantity", "group_code": "Group"},
                      color_discrete_sequence=["#F97316"])
        fig2.update_layout(template="plotly_dark", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("### 🔮 Forecasted Sales by Group (2025)")
        fig3 = px.bar(forecast_group, x="predicted_sales_quantity", y="group_code", orientation="h",
                      labels={"predicted_sales_quantity": "Forecasted Qty", "group_code": "Group"},
                      color_discrete_sequence=["#F97316"])
        fig3.update_layout(template="plotly_dark", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig3, use_container_width=True)

# -------- PAGE: CAT-WISE FORECAST --------
elif menu == "📈 Cat-wise Forecast":
    st.title("📈 Forecast per Tyre Group")

    group_codes = sorted(forecast_df["group_code"].unique())
    selected_group = st.selectbox("Select Group Code", group_codes)

    group_metrics = metrics_df[metrics_df["group_code"] == selected_group].iloc[0]
    forecast = forecast_df[forecast_df["group_code"] == selected_group]
    actuals = train_df[train_df["group_code"] == selected_group][["date", "sales_quantity"]]

    # KPIs
    col1, col2 = st.columns(2)
    col1.metric("MAPE", f"{group_metrics['MAPE']:.2f}%")
    col2.metric("R²", f"{group_metrics['R2']:.2f}")

    # Forecast Plot
    st.markdown(f"### 📅 Actual vs Forecasted Sales for Group {selected_group}")
    fig = px.line(actuals, x="date", y="sales_quantity", labels={"sales_quantity": "Sales"}, title="Historical Sales")
    fig.add_scatter(x=forecast["date"], y=forecast["predicted_sales_quantity"], mode="lines+markers", name="Forecasted")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # Table
    st.markdown("### 📋 Forecast Table")
    st.dataframe(forecast[["date", "predicted_sales_quantity"]].rename(
        columns={"predicted_sales_quantity": "Forecasted Quantity"}).style.format({"Forecasted Quantity": "{:.0f}"}))

# -------- PAGE: RAW DATA VIEW --------
elif menu == "📂 Raw Data View":
    st.title("📂 Full Forecast Data")
    st.dataframe(forecast_df)