import pandas as pd
from scipy.stats import kruskal

# Load the dataset
file_path = "your_dataset.csv"  # Replace with the actual filename
df = pd.read_csv(file_path)

# Checking unique material groups for grouping
material_groups = df['Material_Group'].unique()

# Preparing data for the Kruskal-Wallis test (alpha test)
grouped_data = [df[df['Material_Group'] == group]['Sales_Quantity'].dropna() for group in material_groups]

# Perform the Kruskal-Wallis H-test (non-parametric test for multiple independent samples)
statistic, p_value = kruskal(*grouped_data)

# Output results
print(f"Kruskal-Wallis Test Statistic: {statistic}")
print(f"p-value: {p_value}")

# Interpretation of p-value
if p_value < 0.05:
    print("✅ Statistically significant difference between groups (Good variability).")
else:
    print("⚠️ No significant difference detected (Potential issue with dataset).")
