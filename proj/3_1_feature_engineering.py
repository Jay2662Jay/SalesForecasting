import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# -------- CONFIG --------
TRAIN_PATH = "/Users/jaygamage/Downloads/merged_train_data.csv"
PREDICT_PATH = "/Users/jaygamage/Downloads/merged_prfe_data.csv"
OUTPUT_DIR = "/Users/jaygamage/Downloads/feature_outputs"
TARGET = "sales_quantity"
LAG_MONTHS = list(range(1, 13))  # 1 to 12 months
ROLLING_WINDOWS = [3, 6, 9, 12]
CATEGORICAL_COLS = ["group_code"]  # Only encode group_code

# -------- FEATURE FUNCTIONS --------

def add_time_features(df):
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year
    return df

def add_lag_features(df, group_cols, target, lags):
    df = df.sort_values(by=["date"] + group_cols)
    for lag in lags:
        df[f"{target}_lag_{lag}"] = df.groupby(group_cols)[target].shift(lag)
    return df

def add_rolling_features(df, group_cols, target, windows):
    df = df.sort_values(by=["date"] + group_cols)
    for window in windows:
        df[f"{target}_rollmean_{window}"] = (
            df.groupby(group_cols)[target]
            .shift(1)
            .rolling(window)
            .mean()
            .reset_index(level=0, drop=True)
        )
    return df

def encode_categoricals(train_df, predict_df):
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        full_data = pd.concat([train_df[col], predict_df[col]], axis=0).astype(str)
        le.fit(full_data)
        train_df[col] = le.transform(train_df[col].astype(str))
        predict_df[col] = le.transform(predict_df[col].astype(str))
    return train_df, predict_df

def preprocess(df, is_train=True):
    df["date"] = pd.to_datetime(df["date"])
    df = add_time_features(df)
    if is_train:
        df = add_lag_features(df, group_cols=["group_code"], target=TARGET, lags=LAG_MONTHS)
        df = add_rolling_features(df, group_cols=["group_code"], target=TARGET, windows=ROLLING_WINDOWS)
    return df

# -------- MAIN PIPELINE --------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("📥 Loading data...")
    train_df = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    pred_df = pd.read_csv(PREDICT_PATH, parse_dates=["date"])

    print("🛠️  Applying feature engineering...")
    train_fe = preprocess(train_df, is_train=True)
    predict_fe = preprocess(pred_df, is_train=False)

    print("🏷️  Applying label encoding...")
    train_fe, predict_fe = encode_categoricals(train_fe, predict_fe)

    print("🧹 Filling NaNs in lag/rolling features...")
    required_lags = [f"{TARGET}_lag_{lag}" for lag in LAG_MONTHS]
    required_rolls = [f"{TARGET}_rollmean_{win}" for win in ROLLING_WINDOWS]
    train_fe[required_lags + required_rolls] = train_fe[required_lags + required_rolls].fillna(0)

    train_fe = train_fe.fillna(0)
    predict_fe = predict_fe.fillna(0)

    print("💾 Saving datasets...")
    train_fe.to_csv(f"{OUTPUT_DIR}/train_featured.csv", index=False)
    predict_fe.to_csv(f"{OUTPUT_DIR}/predict_featured.csv", index=False)

    print("✅ Feature engineering complete.")

if __name__ == "__main__":
    main()