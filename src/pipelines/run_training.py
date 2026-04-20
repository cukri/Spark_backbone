from utils.spark_session import get_spark
from models.train_model import train_model
from pyspark.sql.functions import col, mean, sqrt, abs as spark_abs
from pyspark.sql.functions import expm1, log1p
from pyspark.ml.evaluation import RegressionEvaluator

def run_training_pipeline(feature_cols):
    spark = get_spark()

    print("Loading train/test...", flush=True)

    train_df = spark.read.parquet("data/processed/train")
    test_df = spark.read.parquet("data/processed/test")

    print("Training model...", flush=True)

    model, predictions, rmse = train_model(train_df, test_df, feature_cols)

    print("Training finished", flush=True)

    return model, predictions, rmse


def main():

    feature_cols = [

    # ===== LAGS =====
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "lag_24",

    # ===== DELTAS =====
    "delta_1",
    "delta_3",
    "delta_6",
    "delta_12",

    # ===== PERCENT CHANGES =====
    "pct_change_1",
    "pct_change_3",

    # ===== ROLLING AVG =====
    "rolling_avg_3",
    "rolling_avg_6",
    "rolling_avg_12",
    "rolling_avg_24",

    # ===== ROLLING STD =====
    "rolling_std_3",
    "rolling_std_6",
    "rolling_std_12",
    "rolling_std_24",

    # ===== ROLLING MIN/MAX =====
    "rolling_min_6",
    "rolling_max_6",
    "rolling_min_24",
    "rolling_max_24",

    # ===== RANGES =====
    "range_6",
    "range_24",

    # ===== RELATIVE =====
    "ratio_to_avg_6",
    "ratio_to_avg_24",

    # ===== TIME =====
    "hour_sin",
    "hour_cos"
]

    model, predictions, rmse = run_training_pipeline(feature_cols)

    print("\nFeature Importances:")

    for name, imp in sorted(
        zip(feature_cols, model.featureImportances),
        key=lambda x: x[1],
        reverse=True
    ):
     print(f"{name:20s} {imp:.4f}")


    # BASELINE
    baseline_rmse = predictions.select(
    sqrt(mean((col("target") - log1p(col("lag_1")))**2)).alias("rmse")
    ).collect()[0]["rmse"]

    print("\n=== METRICS ===", flush=True)
    print(f"Baseline RMSE (lag_1): {baseline_rmse}", flush=True)
    print(f"Model RMSE: {rmse}", flush=True)

    r2 = RegressionEvaluator(
        labelCol="target",
        predictionCol="prediction",
        metricName="r2"
    ).evaluate(predictions)

    print(f"R2 Score: {r2:.4f}")

    print("\n=== SHOWING PREDICTIONS ===", flush=True)
    predictions = predictions.withColumn(
        "prediction_raw",
        expm1(col("prediction"))
    )

    predictions = predictions.withColumn(
        "target_raw",
        expm1(col("target"))
    )

    predictions = predictions.withColumn(
        "abs_error",
        spark_abs(col("target_raw") - col("prediction_raw"))
    )

    predictions.select(
        "metric_value",
        "target_raw",
        "prediction_raw",
        "abs_error"
    ).show(20, False)

    print("=== END ===", flush=True)

if __name__ == "__main__":
    main()