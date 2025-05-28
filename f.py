#5_c_o
# -------- LIBRARIES --------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import joblib
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------- CONFIGURATION --------
TRAIN_PATH = "/Users/jaygamage/Downloads/pt/feature_outputs/train_featured.csv"  # Historical training data
PREDICT_PATH = "/Users/jaygamage/Downloads/pt/feature_outputs/predict_featured.csv"  # 2025 prediction data
OUTPUT_DIR = "/Users/jaygamage/Downloads/pt/r1.4lg_refined"  # Output path for results
TARGET = "sales_quantity"  # Forecast target
TEST_YEAR = 2024  # Holdout year for validation
N_TRIALS = 100  # Number of Optuna trials
CAT_FEATURES = ["group_code"]  # Categorical feature(s)

# -------- METRIC FUNCTION --------
def get_metrics(y_true, y_pred):
    """Evaluate predictions using MAE, RMSE, MAPE, R² — inverse log1p scale."""
    y_true, y_pred = np.expm1(y_true), np.expm1(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

# -------- PLOT FUNCTION --------
def plot_forecast(dates, y_true, y_pred, group_code, save_path):
    """Plot actual vs predicted sales over time for a group."""
    y_true, y_pred = np.expm1(y_true), np.expm1(y_pred)
    plt.figure(figsize=(10, 5))
    plt.plot(dates, y_true, label="Actual", marker="o")
    plt.plot(dates, y_pred, label="Predicted", marker="x")
    plt.title(f"Forecast for Group {group_code}")
    plt.xlabel("Date")
    plt.ylabel("Sales Quantity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# -------- FEATURE UPDATER FOR RECURSION --------
def update_recursive_features(df):
    """Recalculate lag and rolling features recursively using updated sales."""
    df = df.sort_values("date")
    for lag in [1, 2, 3]:
        df[f"lag_{lag}"] = df[TARGET].shift(lag)
    for window in [3, 6]:
        df[f"rolling_mean_{window}"] = df[TARGET].rolling(window).mean()
    return df

# -------- OPTUNA OBJECTIVE BUILDER --------
def build_objective(train_df, features, cat_features):
    """Build Optuna objective function with expanding window validation."""
    def objective(trial):
        # Hyperparameter space
        params = {
            "iterations": trial.suggest_int("iterations", 300, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 1e-9, 10.0, log=True),
            "loss_function": "RMSE", "random_seed": 42, "verbose": 0
        }

        # Custom penalty coefficients
        alpha, beta = 1000, 1000
        losses = []

        # Expanding window: Train on ≤2022, validate on 2023; then train on ≤2023, validate on 2024
        for train_end, val_year in zip([2022, 2023], [2023, 2024]):
            train_cut = train_df[train_df["date"].dt.year <= train_end]
            val_cut = train_df[train_df["date"].dt.year == val_year]

            X_train, y_train = train_cut[features], np.log1p(train_cut[TARGET])
            X_val, y_val = val_cut[features], np.log1p(val_cut[TARGET])

            model = CatBoostRegressor(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_features, early_stopping_rounds=50)

            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            mape = np.mean(np.abs((np.expm1(y_val) - np.expm1(preds)) / (np.expm1(y_val) + 1e-6))) * 100
            r2 = r2_score(np.expm1(y_val), np.expm1(preds))

            # Custom loss = RMSE + penalty for MAPE > 15 and R² < 0.5
            loss = rmse + alpha * max(0, mape - 15) + beta * max(0.5 - r2, 0)
            losses.append(loss)

        return np.mean(losses)
    return objective

# -------- MAIN EXECUTION --------
def main():
    # Create output directories
    os.makedirs(f"{OUTPUT_DIR}/models", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/forecasts", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)

    # Load train and prediction data
    train_df = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    predict_df = pd.read_csv(PREDICT_PATH, parse_dates=["date"])
    group_codes = train_df["group_code"].unique()

    # Store results
    metric_records = []
    train_metrics = []
    optuna_param_records = []

    # Loop through each group
    for group in group_codes:
        print(f"🔍 Tuning group: {group}")
        train_g = train_df[train_df["group_code"] == group].copy()
        pred_g = predict_df[predict_df["group_code"] == group].copy()

        # Skip groups with no 2025 prediction data
        if pred_g.empty:
            print(f"⚠️ Skipping group {group} — no 2025 data.")
            continue

        # Drop non-feature columns
        drop_cols = ["date", "sales_quantity", "material_group"]
        features = [col for col in train_g.columns if col not in drop_cols and col in pred_g.columns]

        # -------- OPTUNA TUNING --------
        study = optuna.create_study(direction="minimize")
        study.optimize(build_objective(train_g, features, CAT_FEATURES), n_trials=N_TRIALS)
        best_params = study.best_params
        best_params.update({"loss_function": "RMSE", "random_seed": 42, "verbose": 0})
        optuna_param_records.append({**best_params, "group_code": group})

        # -------- FINAL MODEL TRAINING --------
        final_train = train_g[train_g["date"].dt.year <= TEST_YEAR]
        X_train, y_train = final_train[features], np.log1p(final_train[TARGET])
        model = CatBoostRegressor(**best_params)
        model.fit(X_train, y_train, cat_features=CAT_FEATURES)
        joblib.dump(model, f"{OUTPUT_DIR}/models/{group}_catboost_optuna.pkl")

        # -------- VALIDATION METRICS --------
        val_cut = train_g[train_g["date"].dt.year == TEST_YEAR]
        val_preds = model.predict(val_cut[features])
        val_cut["prediction"] = val_preds
        metrics = get_metrics(np.log1p(val_cut[TARGET]), val_preds)
        metrics["group_code"] = group
        metric_records.append(metrics)

        # Plot validation forecast
        plot_forecast(val_cut["date"], np.log1p(val_cut[TARGET]), val_preds, group, f"{OUTPUT_DIR}/plots/{group}_forecast.png")

        # -------- TRAIN METRICS --------
        train_preds = model.predict(X_train)
        train_m = get_metrics(y_train, train_preds)
        train_m["group_code"] = group
        train_metrics.append(train_m)

        # -------- RECURSIVE FORECASTING --------
        # Combine train + predict data to update lags and forecast recursively
        history = pd.concat([train_g, pred_g], ignore_index=True).sort_values("date")
        for idx, row in pred_g.iterrows():
            current_date = row["date"]
            temp_history = history[history["date"] <= current_date].copy()
            temp_history = update_recursive_features(temp_history)
            updated_row = temp_history[temp_history["date"] == current_date]
            pred_features = updated_row[features]
            pred_value = model.predict(pred_features)[0]
            history.loc[history["date"] == current_date, TARGET] = np.expm1(pred_value)  # Store forecasted value

        # Extract 2025 predictions
        pred_g = history[history["date"].dt.year == 2025][["date", "group_code", TARGET]]
        pred_g.rename(columns={TARGET: "predicted_sales_quantity"}, inplace=True)
        pred_g.to_csv(f"{OUTPUT_DIR}/forecasts/{group}_forecast.csv", index=False)

        print(f"✅ Group {group} complete.")

    # -------- SAVE OUTPUT METRICS --------
    pd.DataFrame(metric_records).to_csv(f"{OUTPUT_DIR}/cv_metrics.csv", index=False)
    pd.DataFrame(train_metrics).to_csv(f"{OUTPUT_DIR}/train_metrics.csv", index=False)
    pd.DataFrame(optuna_param_records).to_csv(f"{OUTPUT_DIR}/optuna_params.csv", index=False)
    print("🎯 Forecasting complete for all groups.")

# Entry point
if __name__ == "__main__":
    main()