#2_m_p
import pandas as pd

# Load predicted datasets
promo_2025 = pd.read_csv("/Users/jaygamage/Downloads/pt/adcdata.csv", parse_dates=["date"])
external_2025 = pd.read_csv("/Users/jaygamage/Downloads/pt/ffdata.csv", parse_dates=["date"])

# Load a mapping file that contains group_code + material_group (from training data)
group_map_df = pd.read_csv("/Users/jaygamage/Downloads/feature_outputs/train_featured.csv", usecols=["group_code", "material_group"])
group_map_df = group_map_df.drop_duplicates()

# Merge promo and external features on date only
merged_2025 = pd.merge(
    promo_2025,
    external_2025,
    on=["date"],
    how="left"
)

# Add group_code using mapping
merged_2025 = pd.merge(
    merged_2025,
    group_map_df,
    on="material_group",
    how="left"
)

# Reorder columns to match your required output
merged_2025 = merged_2025[[
    "date", "group_code", "material_group", "advertising_promotion",
    "gdp", "ncpi", "crudeoil", "traffic_index",
    "petrol95", "petrol92", "auto_diesel", "super_diesel", "num_of_holidays"
]]

# ✅ Sort by date and group_code ascending
merged_2025 = merged_2025.sort_values(by=["date", "group_code"]).reset_index(drop=True)

# Save to CSV
output_path = "/Users/jaygamage/Downloads/pt/merged_prfe_data.csv"
merged_2025.to_csv(output_path, index=False)
print(f"✅ Sorted forecast dataset saved as: {output_path}")