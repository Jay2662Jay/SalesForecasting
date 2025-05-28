import pandas as pd

# Load 2025 forecasted promo and external features
promo_2025 = pd.read_csv("/Users/jaygamage/Downloads/adccata.csv", parse_dates=["date"])
external_2025 = pd.read_csv("/Users/jaygamage/Downloads/ffdata.csv", parse_dates=["date"])

# Merge on date, group_code, material_group
merged_2025 = pd.merge(
    promo_2025,
    external_2025,
    on=["date"],
    how="left"
)

# Sort by date and group_code
merged_2025 = merged_2025.sort_values(by=["date", "group_code"]).reset_index(drop=True)

# Save the sorted forecast dataset
merged_2025.to_csv("/Users/jaygamage/Downloads/merged_prfe_dataset.csv", index=False)
print("✅ Sorted forecast dataset saved as 'merged_prfe_dataset.csv'")