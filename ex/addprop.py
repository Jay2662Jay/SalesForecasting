import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
from skopt import gp_minimize
from skopt.space import Real, Categorical
from skopt.utils import use_named_args
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ========== STEP 1: Load and Prepare Data ========== #
file_path = "/Users/jaygamage/Downloads/addata.csv"
df = pd.read_csv(file_path)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Material_Group", "Date"])

# ========== STEP 2: Feature Engineering ========== #
def add_temporal_features(df):
    for lag in [1, 3, 6, 12]:
        df[f"Promo_Lag_{lag}"] = df.groupby("Material_Group")["Advertising_Promotion"].shift(lag)
    for window in [3, 6, 12]:
        df[f"Promo_Rolling_Mean_{window}"] = df.groupby("Material_Group")["Advertising_Promotion"].transform(lambda x: x.rolling(window).mean())
    return df.dropna()

df = add_temporal_features(df)
regressor_cols = [col for col in df.columns if col.startswith("Promo_")]

# ========== STEP 3: Forecasting Per Material_Group ========== #
final_predictions = []
noise_scale = 0.6  # Controls variance: 0 = flat, 1 = full std

for material in df["Material_Group"].unique():
    group_df = df[df["Material_Group"] == material].copy()
    group_df = group_df.sort_values("Date")
    
    if len(group_df) < 36:
        continue  # Skip if not enough history

    prophet_df = group_df.rename(columns={"Date": "ds", "Advertising_Promotion": "y"})

    # Train-validation split
    train_df = prophet_df.iloc[:-12]
    val_df = prophet_df.iloc[-12:]

    # Bayesian Optimization space
    space = [
        Real(0.001, 0.5, name="changepoint_prior_scale"),
        Real(1, 20, name="seasonality_prior_scale"),
        Categorical(["additive", "multiplicative"], name="seasonality_mode")
    ]

    @use_named_args(space)
    def objective(changepoint_prior_scale, seasonality_prior_scale, seasonality_mode):
        model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            seasonality_mode=seasonality_mode,
            interval_width=0.95
        )
        for reg in regressor_cols:
            model.add_regressor(reg)
        model.fit(train_df[["ds", "y"] + regressor_cols])
        forecast = model.predict(val_df[["ds"] + regressor_cols])
        return mean_absolute_error(val_df["y"], forecast["yhat"])

    result = gp_minimize(objective, space, n_calls=20, random_state=42)

    best_params = {
        "changepoint_prior_scale": result.x[0],
        "seasonality_prior_scale": result.x[1],
        "seasonality_mode": result.x[2]
    }

    # Final model with best params on full data
    full_model = Prophet(**best_params, interval_width=0.95)
    for reg in regressor_cols:
        full_model.add_regressor(reg)
    full_model.fit(prophet_df[["ds", "y"] + regressor_cols])

    # ========== STEP 4: Simulate Future Regressors with Mild Variance ========== #
    last_known = group_df.iloc[-12:][regressor_cols]
    mean_vals = last_known.mean()
    std_vals = last_known.std()

    future_dates = pd.date_range(start=group_df["Date"].max() + pd.offsets.MonthBegin(), periods=12, freq="M")

    future_regressors = pd.DataFrame([
        (mean_vals + np.random.normal(0, std_vals * noise_scale)).clip(lower=0)
        for _ in range(12)
    ], columns=regressor_cols)

    future_regressors["ds"] = future_dates

    # Combine historical and future regressor data
    full_future = pd.concat([
        prophet_df[["ds"] + regressor_cols],
        future_regressors
    ], ignore_index=True)

    forecast = full_model.predict(full_future)

    # Get future predictions only
    future_predictions = forecast[forecast["ds"] > group_df["Date"].max()][["ds", "yhat"]].copy()
    future_predictions["Material_Group"] = material
    future_predictions.rename(columns={"ds": "Date", "yhat": "Predicted_Advertising_Promotion"}, inplace=True)

    # Clip predictions: anything ≤ 0 becomes 0
    future_predictions["Predicted_Advertising_Promotion"] = future_predictions["Predicted_Advertising_Promotion"].clip(lower=0)

    final_predictions.append(future_predictions)

    # Optional: Plot forecast
    plt.figure(figsize=(10, 5))
    full_model.plot(forecast)
    plt.title(f"{material} — Forecasted Advertising Promotion")
    plt.tight_layout()
    plt.show()

# ========== STEP 5: Save All Predictions ========== #
final_predictions_df = pd.concat(final_predictions, ignore_index=True)

# Round to 3 decimal places
final_predictions_df["Predicted_Advertising_Promotion"] = final_predictions_df["Predicted_Advertising_Promotion"].round(3)

output_path = "/Users/jaygamage/Downloads/future_promo_forecasts.csv"
final_predictions_df.to_csv(output_path, index=False)
print(f"✅ Saved forecasted Advertising Promotion to: {output_path}")
print(final_predictions_df.head(10))