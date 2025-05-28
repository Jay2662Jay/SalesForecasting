from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, avg
from pyspark.sql.types import StructType, StructField, StringType, FloatType

# Define schema for streaming data (simplified for key pollutants)
schema = StructType([
    StructField("Date", StringType(), True),
    StructField("Time", StringType(), True),
    StructField("CO(GT)", FloatType(), True),
    StructField("PT08.S1(CO)", FloatType(), True),
    StructField("NMHC(GT)", FloatType(), True),
    StructField("C6H6(GT)", FloatType(), True),
    StructField("PT08.S2(NMHC)", FloatType(), True),
    StructField("NOx(GT)", FloatType(), True),
    StructField("PT08.S3(NOx)", FloatType(), True),
    StructField("NO2(GT)", FloatType(), True),
    StructField("PT08.S4(NO2)", FloatType(), True),
    StructField("PT08.S5(O3)", FloatType(), True),
    StructField("T", FloatType(), True),
    StructField("RH", FloatType(), True),
    StructField("AH", FloatType(), True)
])

# Create Spark session
spark = SparkSession.builder.appName("AirQualityStreaming").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Read stream from CSV files
stream_df = spark.readStream \
    .option("sep", ";") \
    .option("header", "true") \
    .schema(schema) \
    .csv("./stream_input")

# Replace missing values (-200) with null
clean_df = stream_df.select([
    when(col(c) == -200, None).otherwise(col(c)).alias(c)
    for c in stream_df.columns
])

# Calculate real-time average of CO and NOx
summary_df = clean_df.groupBy().agg(
    avg("CO(GT)").alias("avg_CO"),
    avg("NOx(GT)").alias("avg_NOx")
)

# Output to console
query = summary_df.writeStream \
    .outputMode("complete") \
    .format("console") \
    .option("truncate", False) \
    .start()

query.awaitTermination()