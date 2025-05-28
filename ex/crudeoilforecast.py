import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

# Load the dataset 
file_path = "/Users/jaygamage/Downloads/dataf_c.csv"  # Update with the path
df = pd.read_csv(file_path)

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Set date as index and sort
df.set_index('date', inplace=True)
df.sort_index(inplace=True)

# Function to forecast crude oil prices
def forecast_prices(df, forecast_periods=12):
    model = ExponentialSmoothing(df['crudeoil'], trend='add', seasonal='add', seasonal_periods=12)
    fitted_model = model.fit()
    forecast = fitted_model.forecast(steps=forecast_periods)
    forecast_df = pd.DataFrame({'date': pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), 
                                                      periods=forecast_periods, freq='M'),
                                'forecasted_crude_oil_price': forecast.values})
    return forecast_df

# Function to plot the forecast
def plot_forecast(df, forecast_df):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['crudeoil'], label='Actual Prices', marker='o')
    plt.plot(forecast_df['date'], forecast_df['forecasted_crude_oil_price'], 
             label='Forecasted Prices', linestyle='dashed', marker='o', color='red')
    plt.xlabel('Date')
    plt.ylabel('Crude Oil Price')
    plt.title('Crude Oil Price Forecast for Next 12 Months')
    plt.legend()
    plt.grid()
    plt.show()

# Perform forecasting
forecast_df = forecast_prices(df, forecast_periods=12)

# Display the forecasted data
print(forecast_df)

# Plot the forecast
plot_forecast(df, forecast_df)
