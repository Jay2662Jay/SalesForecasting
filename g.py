#6_d_s
import streamlit as st
import pandas as pd
import plotly.express as px
import os
from glob import glob

# -------- CONFIG --------
FORECAST_DIR = "/Users/jaygamage/Downloads/pt/r1.4lg_refined/forecasts"
METRICS_PATH = "/Users/jaygamage/Downloads/pt/r1.4lg_refined/train_metrics.csv"
TRAIN_PATH = "/Users/jaygamage/Downloads/pt/feature_outputs/train_featured.csv"

# Group mapping
group_map = {
    0: "Motor Cycle TT", 1: "Motor Cycle TL", 2: "Scooter TT", 3: "Scooter TL",
    4: "3 Wheeler", 5: "Light Truck", 6: "Truck /Bus", 7: "Industrial",
    8: "Grader", 9: "Agri Front", 10: "Agri Rear", 11: "Radial car",
    12: "Radial van", 13: "Radial SUV", 14: "LTR", 15: "TBR",
    16: "Trad. Tube", 17: "Trad.Grader", 18: "Trad. Radial Car",
    19: "Trad. Radial SUV", 20: "Trad. LTR", 21: "Trad. TBR", 22: "Trad. Flap"
}

st.set_page_config(page_title="Tyre Sales Forecast", layout="wide")

# -------- STYLE --------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)),
                    url("https://i.imgur.com/buIPTT5.jpeg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .block-container {
        background-color: rgba(0, 0, 0, 0.2);
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .css-1d391kg, .css-18ni7ap, .stDataFrame, .stMetric {
        background-color: rgba(255, 255, 255, 0.06) !important;
        border-radius: 0.5rem;
        padding: 0.75rem;
    }
    .stMarkdown, .stRadio, .stSelectbox, .stSlider, .stMetric {
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------- LOAD DATA --------
@st.cache_data
def load_data():
    metrics = pd.read_csv(METRICS_PATH)
    train = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    forecast_files = glob(os.path.join(FORECAST_DIR, "*_forecast.csv"))
    forecast_list = [pd.read_csv(f, parse_dates=["date"]) for f in forecast_files]
    forecast_df = pd.concat(forecast_list, ignore_index=True)

    train["group_name"] = train["group_code"].map(group_map)
    forecast_df["group_name"] = forecast_df["group_code"].map(group_map)
    metrics["group_name"] = metrics["group_code"].map(group_map)

    return metrics, train, forecast_df

metrics_df, train_df, forecast_df = load_data()

# -------- SIDEBAR NAVIGATION --------
from PIL import Image
logo = Image.open("/Users/jaygamage/Downloads/JAY.jpg")
st.sidebar.image(logo, use_container_width=True)
st.markdown(
    """
    <style>
    /* Make sidebar image circular */
    [data-testid="stSidebar"] img {
        border-radius: 50%;
        border: 2px solid white;
        padding: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown("# JAY Tyre's Sales Dashboard")
menu = st.sidebar.radio("Navigation", [
    "📊 Dashboard",
    "📈 Cat-wise Forecast",
    "📂 Raw Data View",
    "📑 Feature Data View"
])

# -------- PAGE: DASHBOARD --------
if menu == "📊 Dashboard":
    st.title("Sales Forecast Overview")

    monthly_sales = train_df.copy()
    monthly_sales["month"] = monthly_sales["date"].dt.month
    monthly_sales["year"] = monthly_sales["date"].dt.year
    available_years = ["ALL"] + sorted(monthly_sales["year"].unique().tolist() + [2025])
    selected_year = st.sidebar.selectbox("📅 Select Year", available_years, index=available_years.index(2024))

    if selected_year == "ALL":
        # Actual sales per year
        actual_yearly = train_df.copy()
        actual_yearly["year"] = actual_yearly["date"].dt.year
        actual_totals = actual_yearly.groupby("year")["sales_quantity"].sum().reset_index()
        actual_totals["type"] = "Actual"

        # Forecasted sales (2025)
        forecast_df["year"] = forecast_df["date"].dt.year
        forecast_totals = forecast_df.groupby("year")["predicted_sales_quantity"].sum().reset_index()
        forecast_totals.rename(columns={"predicted_sales_quantity": "sales_quantity"}, inplace=True)
        forecast_totals["type"] = "Forecast"

        # Combine
        all_sales = pd.concat([actual_totals, forecast_totals])
        st.markdown("### 📊 All Sales")

        fig1 = px.bar(
            all_sales,
            x="year", y="sales_quantity", color="type", barmode="group",
            labels={"sales_quantity": "Total Sales", "year": "Year"},
            text="sales_quantity"
        )
        fig1.update_layout(
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig1, use_container_width=True)

    else:
        if selected_year == 2025:
            forecast_df["month"] = forecast_df["date"].dt.month
            forecast_summary = forecast_df.groupby("month")["predicted_sales_quantity"].sum().reset_index()
            forecast_summary.rename(columns={"predicted_sales_quantity": "sales_quantity"}, inplace=True)
            monthly_summary = forecast_summary
        else:
            filtered_sales = monthly_sales[monthly_sales["year"] == selected_year]
            monthly_summary = filtered_sales.groupby("month")["sales_quantity"].sum().reset_index()

        st.markdown(f"### 📆 {'Forecasted' if selected_year == 2025 else 'Actual'} Sales Order by Month ({selected_year})")
        fig1 = px.line(monthly_summary, x="month", y="sales_quantity", markers=True)
        fig1.update_layout(
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig1, use_container_width=True)
    # === Monthly or All sales chart finishes here ===

    # Always display these two sections
    top_sellers = (
        train_df.groupby("group_name")["sales_quantity"]
        .sum()
        .reset_index()
        .sort_values("sales_quantity", ascending=False)
        .head(7)
    )
    category_order_top = top_sellers["group_name"].tolist()[::-1]

    fig2 = px.bar(top_sellers, x="sales_quantity", y="group_name", orientation="h", text="sales_quantity")
    fig2.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(categoryorder="array", categoryarray=category_order_top, title=None)
    )
    fig2.update_traces(texttemplate='%{text:.0f}', textposition='outside')

    forecast_group = (
        forecast_df.groupby("group_name")["predicted_sales_quantity"]
        .sum()
        .reset_index()
        .sort_values("predicted_sales_quantity", ascending=False)
        .head(5)
    )
    category_order_forecast = forecast_group["group_name"].tolist()[::-1]

    fig3 = px.bar(forecast_group, x="predicted_sales_quantity", y="group_name", orientation="h", text="predicted_sales_quantity")
    fig3.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(categoryorder="array", categoryarray=category_order_forecast, title=None)
    )
    fig3.update_traces(texttemplate='%{text:.0f}', textposition='outside')

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Top Selling Tyre Groups")
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.markdown("### Forecasted Sales by Group (2025)")
        st.plotly_chart(fig3, use_container_width=True)
        

    # -------- PAGE: CAT-WISE FORECAST --------
elif menu == "📈 Cat-wise Forecast":
        st.title("Forecast per Tyre Group")

        group_names = sorted(forecast_df["group_name"].unique())
        selected_group = st.selectbox("Select Tyre Group", group_names)

        group_metrics = metrics_df[metrics_df["group_name"] == selected_group].iloc[0]
        forecast_all = forecast_df[forecast_df["group_name"] == selected_group]
        actuals = train_df[train_df["group_name"] == selected_group][["date", "sales_quantity"]]

        forecast_all["month"] = forecast_all["date"].dt.month
        available_months = sorted(forecast_all["month"].unique())
        default_month_index = min(3, len(available_months))
        num_months = st.slider("Number of Forecast Months to Display", 1, len(available_months), default_month_index)
        forecast = forecast_all[forecast_all["month"] <= available_months[num_months - 1]]

        st.metric("MAPE", f"{group_metrics['MAPE']:.2f}%")

        st.markdown(f"### 📅 Actual vs Forecasted Sales for {selected_group}")
        fig = px.line(actuals, x="date", y="sales_quantity", title="Historical Sales")
        fig.add_scatter(x=forecast["date"], y=forecast["predicted_sales_quantity"], mode="lines+markers", name="Forecasted")
        fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📋 Forecast Table")
        st.dataframe(forecast[["date", "predicted_sales_quantity"]].rename(
        columns={"predicted_sales_quantity": "Forecasted Quantity"}).style.format({"Forecasted Quantity": "{:.0f}"}))

    # -------- PAGE: RAW DATA VIEW --------
elif menu == "📂 Raw Data View":
        st.title("Raw Forecast Data")

        st.markdown("### 📈 Total Sales Over Time")
        sales_over_time = train_df.groupby("date")["sales_quantity"].sum().reset_index()
        fig = px.line(sales_over_time, x="date", y="sales_quantity", title="Sales Quantity Over Time", markers=True)
        fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📉 Group-wise Sales Trend")
        group_names = sorted(train_df["group_name"].dropna().unique())
        selected_group = st.selectbox("Select Tyre Group", group_names)
        group_data = train_df[train_df["group_name"] == selected_group]

        fig_group = px.line(group_data, x="date", y="sales_quantity", title=f"Sales Trend for {selected_group}", markers=True)
        fig_group.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_group, use_container_width=True)

        st.markdown("### 📋 Forecast Data Table")
        st.dataframe(forecast_df)

# -------- PAGE: FEATURE DATA VIEW --------
elif menu == "📑 Feature Data View":
    st.title("Feature Data View")

    actual_features = pd.read_csv("/Users/jaygamage/Downloads/dataf_c.csv", parse_dates=["date"])
    features_pred = pd.read_csv("/Users/jaygamage/Downloads/ffdata.csv", parse_dates=["date"])

    for df in [actual_features, features_pred]:
        if "group_code" in df.columns:
            df["group_name"] = df["group_code"].map(group_map)

    shared_features = list(
        set(actual_features.select_dtypes(include='number').columns) &
        set(features_pred.select_dtypes(include='number').columns)
    )
    shared_features = [col for col in shared_features if col not in ["group_code"]]

    selected_feature = st.selectbox("Select Feature", shared_features)

    if "group_name" in actual_features.columns:
        groups = actual_features["group_name"].dropna().unique()
        selected_group = st.selectbox("Filter by Group", sorted(groups))
        act = actual_features[actual_features["group_name"] == selected_group]
        pred = features_pred[features_pred["group_name"] == selected_group]
    else:
        act = actual_features
        pred = features_pred

    st.markdown(f"### 📈 {selected_feature} — Actual vs Predicted")
    fig = px.line(act, x="date", y=selected_feature, title="Actual Feature")
    fig.add_scatter(x=pred["date"], y=pred[selected_feature], mode="lines", name="Predicted")
    fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📋 Combined Table View")
    merged = pd.merge(
        act[["date", selected_feature]],
        pred[["date", selected_feature]],
        on="date", how="outer", suffixes=("_actual", "_predicted")
    )
    st.dataframe(merged.sort_values("date").reset_index(drop=True))