import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import mstats

# Load dataset
file_path = "/Users/jaygamage/Downloads/addata.csv"
df = pd.read_csv(file_path)

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# **Identify categorical columns**
categorical_cols = ["Material_Group"]

# **Convert categorical columns properly**
df[categorical_cols] = df[categorical_cols].astype("category")

# Feature Engineering - Adding Lag Features
for lag in [1, 3, 6, 12]:
    df[f"Promo_Lag_{lag}"] = df.groupby("Material_Group")["Advertising_Promotion"].shift(lag)

# Feature Engineering - Adding Moving Averages
for window in [3, 6, 12]:
    df[f"Promo_Rolling_Mean_{window}"] = df.groupby("Material_Group")["Advertising_Promotion"].transform(lambda x: x.rolling(window=window).mean())

# **Fourier Features for Seasonality**
def compute_fourier_terms(series, n_terms=3):
    t = np.arange(len(series))
    fourier_terms = [np.sin(2 * np.pi * (i+1) * t / len(series)) for i in range(n_terms)]
    return np.array(fourier_terms).T  # Transpose to align with dataframe structure

fourier_features = df.groupby("Material_Group")["Advertising_Promotion"].apply(lambda x: compute_fourier_terms(x))
fourier_df = pd.DataFrame(np.vstack(fourier_features), columns=[f"Fourier_{i}" for i in range(1, 4)])
df = df.reset_index(drop=True)
df = pd.concat([df, fourier_df], axis=1)

# **Interaction Features (Lag * Moving Average)**
df["Interaction_3"] = df["Promo_Lag_3"] * df["Promo_Rolling_Mean_3"]
df["Interaction_6"] = df["Promo_Lag_6"] * df["Promo_Rolling_Mean_6"]
df["Interaction_12"] = df["Promo_Lag_12"] * df["Promo_Rolling_Mean_12"]

# **Momentum Indicator (Rate of Change)**
df["Rate_of_Change"] = df.groupby("Material_Group")["Advertising_Promotion"].pct_change()

# **Handle Outliers using Winsorization**
for col in ["Advertising_Promotion"] + [f"Promo_Lag_{i}" for i in [1, 3, 6, 12]] + \
           [f"Promo_Rolling_Mean_{i}" for i in [3, 6, 12]] + ["Rate_of_Change"]:
    df[col] = mstats.winsorize(df[col], limits=[0.05, 0.05])  # Capping outliers at 5%

# Drop NaN values due to lagging
df = df.dropna()

# Define Hyperparameter Space for Bayesian Optimization
space = [
    Integer(500, 1200, name="iterations"),
    Integer(3, 10, name="depth"),
    Real(0.005, 0.3, name="learning_rate"),
    Real(0.5, 15, name="l2_leaf_reg"),
    Real(0.5, 1.0, name="subsample"),
    Real(0.5, 1.0, name="colsample_bylevel")
]

# Store results
catboost_results = []

# Train and optimize CatBoost for each Material Group
for material in df["Material_Group"].unique():
    category_df = df[df["Material_Group"] == material].copy()

    # Define Features & Target
    X = category_df.drop(columns=["Date", "Advertising_Promotion", "Group_Code"])
    y = category_df["Advertising_Promotion"]

    # **Ensure categorical feature indices are correctly identified**
    categorical_features = [X.columns.get_loc(col) for col in categorical_cols if col in X.columns]

    # Split data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # **Convert categorical features to category codes**
    for col in categorical_cols:
        X_train[col] = X_train[col].astype("category").cat.codes
        X_test[col] = X_test[col].astype("category").cat.codes

    # **Create CatBoost Pool for Categorical Features**
    train_pool = Pool(X_train, y_train, cat_features=categorical_features)
    test_pool = Pool(X_test, y_test, cat_features=categorical_features)

    # Define Objective Function for Bayesian Optimization
    @use_named_args(space)
    def objective(iterations, depth, learning_rate, l2_leaf_reg, subsample, colsample_bylevel):
        model = CatBoostRegressor(
            iterations=iterations, depth=depth, learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg, subsample=subsample, colsample_bylevel=colsample_bylevel,
            loss_function="RMSE", verbose=0
        )
        model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=20, verbose=False)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        return mae  # Minimize MAE

    # Perform Bayesian Optimization
    result = gp_minimize(objective, space, n_calls=60, random_state=42)  # Increased `n_calls` for deeper search

    # Extract Best Hyperparameters
    best_params = {
        "iterations": result.x[0],
        "depth": result.x[1],
        "learning_rate": result.x[2],
        "l2_leaf_reg": result.x[3],
        "subsample": result.x[4],
        "colsample_bylevel": result.x[5]
    }

    # Train CatBoost with Optimized Parameters
    best_catboost = CatBoostRegressor(**best_params, loss_function="RMSE", verbose=0)
    best_catboost.fit(train_pool, eval_set=test_pool, early_stopping_rounds=20, verbose=False)

    # Make Predictions
    y_pred = best_catboost.predict(X_test)

    # Evaluate Optimized Model
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Store Results
    catboost_results.append([material, best_params["iterations"], best_params["depth"],
                             best_params["learning_rate"], best_params["l2_leaf_reg"], 
                             best_params["subsample"], best_params["colsample_bylevel"], mae, rmse, r2])

    # Plot Results
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.index, y_test.values, label="Actual", marker="o", linestyle="--")
    plt.plot(y_test.index, y_pred, label="Optimized CatBoost Prediction", marker="s", linestyle="-")
    plt.title(f"Optimized CatBoost Forecast for {material}")
    plt.xlabel("Date")
    plt.ylabel("Advertising Promotion Spend")
    plt.legend()
    plt.show()

# Convert results to DataFrame
catboost_optimized_df = pd.DataFrame(catboost_results, columns=["Material_Group", "Best_Iterations", 
                                                                "Best_Depth", "Best_Learning_Rate", 
                                                                "Best_L2_Reg", "Best_Subsample",
                                                                "Best_Colsample_ByLevel", "MAE", "RMSE", "R²"])

# Display evaluation results
catboost_optimized_df