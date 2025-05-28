#2_m_a
import pandas as pd

# Load datasets
sales = pd.read_csv("/Users/jaygamage/Downloads/pt/rdata.csv", parse_dates=["date"])
external = pd.read_csv("/Users/jaygamage/Downloads/pt/dataf_c.csv", parse_dates=["date"])
promo = pd.read_csv("/Users/jaygamage/Downloads/pt/addata.csv", parse_dates=["date"])

# 🔁 Correct mapping from material_group to group_code
group_mapping = {
    "Motor Cycle TT": 0, "Motor Cycle TL": 1, "Scooter TT": 2, "Scooter TL": 3,
    "3 Wheeler": 4, "Light Truck": 5, "Truck /Bus": 6, "Industrial": 7,
    "Grader": 8, "Agri Front": 9, "Agri Rear": 10, "Radial car": 11,
    "Radial van": 12, "Radial SUV": 13, "LTR": 14, "TBR": 15,
    "Trad. Tube": 16, "Trad.Grader": 17, "Trad. Radial Car": 18,
    "Trad. Radial SUV": 19, "Trad. LTR": 20, "Trad. TBR": 21, "Trad. Flap": 22
}

# Step 1: Merge sales with promo on date + material_group (remove old group_code if any)
sales = sales.drop(columns=["group_code"], errors="ignore")
promo = promo.drop(columns=["group_code"], errors="ignore")
merged = pd.merge(sales, promo, on=["date", "material_group"], how="left")

# Step 2: Merge with external features on date
final_merged = pd.merge(merged, external, on="date", how="left")

# Step 3: Assign correct group_code from material_group
final_merged["group_code"] = final_merged["material_group"].map(group_mapping)

# ✅ Step 4: Move group_code to second column
cols = list(final_merged.columns)
cols.insert(1, cols.pop(cols.index("group_code")))
final_merged = final_merged[cols]

# Step 5: Sort by date and group_code
final_merged = final_merged.sort_values(by=["date", "group_code"]).reset_index(drop=True)

# Step 6: Save merged dataset
final_merged.to_csv("/Users/jaygamage/Downloads/pt/merged_train_data.csv", index=False)
print("✅ Merged dataset saved at /Users/jaygamage/Downloads/pt/merged_train_data.csv")

# Optional preview
print(final_merged.head())