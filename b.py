#1_f
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# === Load data ===
df = pd.read_csv("/Users/jaygamage/Downloads/pt/dataf_c.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date").asfreq("M")
df_base = df.copy()  # backup for GDP
df = df.drop(columns=["gdp", "num_of_holidays"])

# === Define SARIMA parameters ===
sarima_params = {
    "petrol95":      {"order": (1, 2, 2), "seasonal_order": (0, 0, 0, 12)},
    "auto_diesel":   {"order": (0, 1, 0), "seasonal_order": (0, 0, 0, 12)},
    "traffic_index": {"order": (1, 2, 2), "seasonal_order": (1, 1, 0, 12)},
}

# === Custom crude oil forecast ===
crude_forecast_custom = [
    72.85, 73.47, 73.68, 71.29, 70.71, 68.39,
    68.65, 69.68, 70.98, 72.79, 73.69, 72.74
]

# === Forecast function ===
def forecast_sarima(series, order, seasonal_order, steps=12):
    model = sm.tsa.statespace.SARIMAX(
        series, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False
    )
    results = model.fit(disp=False)
    return results.forecast(steps=steps)

# === Forecast horizon setup ===
forecast_horizon = 12
future_index = pd.date_range(start=df.index[-1] + pd.offsets.MonthBegin(1),
                             periods=forecast_horizon, freq="M")
all_forecasts = pd.DataFrame(index=future_index)

# === Main forecasting loop ===
for col in df.columns:
    series = df[col].dropna()

    if col == "super_diesel":
        target = 331
        pred = [target] + [target + np.random.uniform(-2, 2) for _ in range(11)]

    elif col == "petrol92":
        target = 309
        pred = [target] + [target + np.random.uniform(-2, 2) for _ in range(11)]

    elif col == "crudeoil":
        pred = crude_forecast_custom

    elif col == "ncpi":
        slope = max((series[-1] - series[-13]) / 12, 0.5)
        last_val = series.iloc[-1]
        pred = [last_val + slope * (i + 1) for i in range(forecast_horizon)]

    elif col == "petrol95":
        params = sarima_params[col]
        raw_pred = forecast_sarima(series, params["order"], params["seasonal_order"], forecast_horizon)
        pred = [max(350, val) for val in raw_pred]

    else:
        params = sarima_params[col]
        pred = forecast_sarima(series, params["order"], params["seasonal_order"], forecast_horizon)

    all_forecasts[col] = pd.Series(pred, index=future_index)

# === GDP via Quarterly ARIMA → Monthly Mapping ===
df_gdp = df_base[['gdp']].copy()
df_gdp['date'] = df_gdp.index
df_q = df_gdp.resample('Q', on='date').mean()

gdp_model = ARIMA(df_q['gdp'], order=(1, 1, 1))
gdp_model_fit = gdp_model.fit()
forecast_qtr = gdp_model_fit.forecast(steps=4)
forecast_qtr.index = pd.date_range(start=df_q.index[-1] + pd.offsets.QuarterEnd(), periods=4, freq='Q')

# Expand quarterly GDP into months
qtr_to_month_map = {}
for qtr_date, gdp_val in zip(forecast_qtr.index, forecast_qtr.values):
    q_months = pd.date_range(start=qtr_date - pd.offsets.QuarterEnd(startingMonth=3) + pd.offsets.MonthBegin(), periods=3, freq='MS')
    for m in q_months:
        qtr_to_month_map[m.strftime("%Y-%m")] = gdp_val

monthly_index = all_forecasts.index
gdp_forecast = [qtr_to_month_map.get(date.strftime("%Y-%m"), np.nan) for date in monthly_index]
all_forecasts['gdp'] = gdp_forecast

# === Add number of holidays manually ===
holidays_per_month = {
    1: 2, 2: 3, 3: 3, 4: 5, 5: 3, 6: 2,
    7: 1, 8: 1, 9: 2, 10: 2, 11: 1, 12: 2
}
all_forecasts['num_of_holidays'] = all_forecasts.index.month.map(holidays_per_month)

# === Add date column and round values ===
all_forecasts = all_forecasts.reset_index().rename(columns={"index": "date"})
all_forecasts = all_forecasts.round(2)

# === Final output ===
print("✅ Final Forecasts (12 Months):")
print(all_forecasts.round(2))

# === Save to CSV ===
all_forecasts.to_csv('/Users/jaygamage/Downloads/pt/ffdata.csv', index=False)
print("💾 Saved to: /Users/jaygamage/Downloads/pt/ffdata.csv")