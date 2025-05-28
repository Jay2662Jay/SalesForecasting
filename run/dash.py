import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load Data
data_cleaned = pd.read_csv('/Users/jaygamage/Downloads/data1.csv')
data_cleaned['date'] = pd.to_datetime(data_cleaned['date'])
data_cleaned = data_cleaned.set_index('date')

# Load Models
sarima_fit = joblib.load("sarima_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")

# Forecast Sales
def forecast_sales(data, forecast_period=12):
    """Generate SARIMA + XGBoost hybrid forecast"""
    sarima_pred = sarima_fit.forecast(steps=forecast_period)
    future_dates = pd.date_range(start=data.index[-1], periods=forecast_period + 1, freq='M')[1:]

    # Create future features for XGBoost
    future_features = pd.DataFrame({
        'year': future_dates.year,
        'month': future_dates.month,
        'quarter': future_dates.quarter,
        'gdp': data['gdp'].iloc[-1]  # Assuming the last known GDP value is used for future predictions
    }, index=future_dates)

    xgb_residual_pred = xgb_model.predict(future_features)
    
    # Hybrid Model Forecast
    hybrid_forecast = sarima_pred + xgb_residual_pred
    return future_dates, hybrid_forecast, sarima_pred

# Streamlit App UI
st.title("📊 Tyre Sales Forecasting Dashboard")

# Forecast Period Selector
forecast_period = st.slider("Select Forecast Period (Months):", min_value=1, max_value=24, value=12)

# Generate Forecast
future_dates, hybrid_forecast, sarima_pred = forecast_sales(data_cleaned, forecast_period)

# Plot Forecast
fig = go.Figure()

# Add actual sales data
fig.add_trace(go.Scatter(x=data_cleaned.index, y=data_cleaned['sales quantity'], mode='lines', name='Actual Sales'))

# Add SARIMA forecast
fig.add_trace(go.Scatter(x=future_dates, y=sarima_pred, mode='lines', name='SARIMA Forecast'))

# Add Hybrid forecast
fig.add_trace(go.Scatter(x=future_dates, y=hybrid_forecast, mode='lines', name='Hybrid Forecast'))

# Update layout
fig.update_layout(
    title="Sales Forecast",
    xaxis_title="Date",
    yaxis_title="Sales Quantity",
    legend_title="Legend",
    template="plotly_white"
)

# Display plot in Streamlit
st.plotly_chart(fig)