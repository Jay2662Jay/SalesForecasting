import pandas as pd
import snowflake.connector

# Step 1: Load your cleaned CSV file
data_path = "cleaned_air_quality.csv"  # Ensure this is a CSV version of your cleaned data
df = pd.read_csv(data_path)

# Step 2: Snowflake connection config
conn = snowflake.connector.connect(
    user="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    account="YOUR_ACCOUNT_ID",  # e.g., xy12345.ap-southeast-1
    warehouse="YOUR_WAREHOUSE",
    database="YOUR_DATABASE",
    schema="YOUR_SCHEMA"
)

cs = conn.cursor()

# Step 3: Create table
create_table_query = """
CREATE OR REPLACE TABLE AIR_QUALITY (
    CO_GT FLOAT,
    NOx_GT FLOAT,
    NO2_GT FLOAT,
    T FLOAT,
    RH FLOAT,
    AH FLOAT
);
"""
cs.execute(create_table_query)

# Step 4: Upload data row by row (simple method)
insert_query = """
INSERT INTO AIR_QUALITY (CO_GT, NOx_GT, NO2_GT, T, RH, AH)
VALUES (%s, %s, %s, %s, %s, %s)
"""
for _, row in df.iterrows():
    cs.execute(insert_query, (
        row["CO(GT)"],
        row["NOx(GT)"],
        row["NO2(GT)"],
        row["T"],
        row["RH"],
        row["AH"]
    ))

# Step 5: Query average values
cs.execute("""
    SELECT 
        AVG(CO_GT) AS avg_co,
        AVG(NOx_GT) AS avg_nox,
        AVG(NO2_GT) AS avg_no2
    FROM AIR_QUALITY
""")

print("Air Quality Averages:")
for row in cs.fetchall():
    print(row)

# Close connection
cs.close()
conn.close()