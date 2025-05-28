import pandas as pd
import time
import os

# Path to the cleaned CSV (if needed, convert from parquet first)
source_csv = "/Users/jaygamage/Downloads/AirQualityUCI.csv"
output_dir = "./stream_input"

# Load the original data
df = pd.read_csv(source_csv, sep=';', low_memory=False)

# Drop unwanted columns and NaNs
df = df.drop(columns=["Unnamed: 15", "Unnamed: 16"], errors='ignore')
df = df.dropna(subset=["Date", "Time"])

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Write one row at a time as a new CSV file every second
for i, row in df.iterrows():
    file_path = os.path.join(output_dir, f"row_{i}.csv")
    row.to_frame().T.to_csv(file_path, index=False, sep=';')
    print(f"Streamed row {i} to {file_path}")
    time.sleep(1)