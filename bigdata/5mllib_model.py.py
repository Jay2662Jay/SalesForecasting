from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Start Spark session
spark = SparkSession.builder.appName("AirQualityML").getOrCreate()

# Load the cleaned parquet or CSV file
df = spark.read.parquet("cleaned_air_quality.parquet")

# Drop rows where target (CO) is null
df = df.filter(df["CO(GT)"].isNotNull())

# Features to use for prediction
feature_cols = ["NOx(GT)", "NO2(GT)", "T", "RH", "AH"]

# Drop rows with nulls in features
for col_name in feature_cols:
    df = df.filter(df[col_name].isNotNull())

# Assemble features into vector
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df).select("features", col("CO(GT)").alias("label"))

# Train-test split
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Model: Random Forest Regressor
model = RandomForestRegressor(featuresCol="features", labelCol="label")
rf_model = model.fit(train_df)

# Predictions
predictions = rf_model.transform(test_df)

# Evaluation
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2").evaluate(predictions)

# Output results
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"R² Score: {r2:.3f}")

# Show sample predictions
predictions.select("prediction", "label", "features").show(10, truncate=False)