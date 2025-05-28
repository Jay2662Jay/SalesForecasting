from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, to_timestamp, when

# Start Spark session
spark = SparkSession.builder \
    .appName("AirQualityPreprocessing") \
    .getOrCreate()

# Load CSV file
df = spark.read.csv("/Users/jaygamage/Downloads/AirQualityUCI.csv", sep=';', header=True, inferSchema=True)

# Drop empty last column and unnamed columns
df = df.drop("_c15", "_c16")  # drop unnamed empty columns if exist

# Replace commas with dots (European decimal) and convert to float
for col_name in df.columns:
    df = df.withColumn(col_name, regexp_replace(col(col_name).cast("string"), ",", "."))
    df = df.withColumn(col_name, col(col_name).cast("float"))

# Combine Date and Time into a timestamp column
df = df.withColumn("timestamp", to_timestamp(col("Date") + " " + col("Time"), "dd/MM/yyyy HH.mm.ss"))

# Drop original Date and Time
df = df.drop("Date", "Time")

# Handle missing values: replace -200 with null
df = df.select([when(col(c) == -200, None).otherwise(col(c)).alias(c) for c in df.columns])

# Show cleaned data
df.show(5, truncate=False)

# Save cleaned data to parquet (optional)
df.write.mode("overwrite").parquet("cleaned_air_quality.parquet")