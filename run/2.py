import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Load Data (make sure the dataset is available in your working directory)
@st.cache
def load_data():
    df = pd.read_csv('/Users/jaygamage/Downloads/data.csv')  # Replace with the actual filename and path
    print(df.columns)  # Debugging line to print columns
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

df = load_data()

# Sidebar Configuration
st.sidebar.title("Tyre Sales Forecasting")
forecast_period = st.sidebar.slider("Select forecast period (months)", 1, 24, 12)

# Main Title
st.title("📊 Tyre Sales Forecasting System")

# Display raw data
st.subheader("📅 Raw Sales Data")
st.dataframe(df.head())

# Plot Sales Trend
st.subheader("📈 Sales Trend Over Time")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df.index, df['sales_quantity'], marker='o', linestyle='-')  # Adjust 'sales_quantity' to your column name for sales
ax.set_xlabel("Date")
ax.set_ylabel("Total Sales")
ax.set_title("Monthly Sales Trend")
ax.grid(True)
st.pyplot(fig)

# Prepare Data for Prophet
df_prophet = df.reset_index()[['date', 'sales_quantity']]  # Adjust 'date' and 'sales_quantity' to your columns
df_prophet.columns = ['ds', 'y']  # Prophet requires columns 'ds' (date) and 'y' (value)

# Train Prophet Model
model = Prophet()
model.fit(df_prophet)

# Forecast Future Sales
future = model.make_future_dataframe(periods=forecast_period * 30, freq='D')
forecast = model.predict(future)

# Display Forecast
st.subheader(f"🔮 {forecast_period}-Month Sales Forecast")
fig_forecast = model.plot(forecast)
st.pyplot(fig_forecast)

# Show Forecast Data
st.subheader("📋 Forecasted Data")
st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_period))

# Closing Message
st.success("🚀 Dashboard is running successfully! Modify forecast period using the slider.")