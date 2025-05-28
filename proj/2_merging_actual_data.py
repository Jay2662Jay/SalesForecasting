import pandas as pd

# Load datasets
sales = pd.read_csv("/Users/jaygamage/Downloads/rdata.csv", parse_dates=["date"])
external = pd.read_csv("/Users/jaygamage/Downloads/dataf_c.csv", parse_dates=["date"])
promo = pd.read_csv("/Users/jaygamage/Downloads/addata.csv", parse_dates=["date"])

# Step 1: Merge sales with promo on date + group_code + material_group
merged = pd.merge(sales, promo, on=["date", "group_code", "material_group"], how="left")

# Step 2: Merge with external features on date
final_merged = pd.merge(merged, external, on="date", how="left")

# Step 3: Save merged dataset
final_merged.to_csv("/Users/jaygamage/Downloads/merged_train_data.csv", index=False)
print("✅ Merged dataset saved at /Users/jaygamage/Downloads/merged_train_data.csv")

# Optional preview
print(final_merged.head())