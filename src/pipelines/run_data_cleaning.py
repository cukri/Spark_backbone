from utils.spark_session import get_spark
from features.data_cleaning import clean_metrics


def run():
    spark = get_spark()

    print("=== DATA CLEANING PIPELINE ===", flush=True)

    # ======================
    # LOAD RAW DATA
    # ======================
    print("\nLoading raw data...", flush=True)

    df = spark.read.parquet("data/raw/metrics")

    print(f"Rows before cleaning: {df.count():,}", flush=True)

    # ======================
    # CLEANING
    # ======================
    print("\nRunning cleaning module...", flush=True)

    cleaned_df = clean_metrics(df)

    print(f"Rows after cleaning: {cleaned_df.count():,}", flush=True)

    # ======================
    # PREVIEW
    # ======================
    print("\n=== CLEANED DATA PREVIEW ===", flush=True)

    cleaned_df.select(
        "event_time",
        "available_at",
        "entity_id",
        "metric_type",
        "metric_value",
        "metric_value_clipped",
        "metric_value_log"
    ).show(10, truncate=False)

    # ======================
    # SAVE OUTPUT
    # ======================
    print("\nSaving cleaned data...", flush=True)

    cleaned_df.write.mode("overwrite").parquet("data/cleaned/metrics")

    print("Saved to: data/cleaned/metrics", flush=True)

    print("\n=== PIPELINE FINISHED ===", flush=True)


if __name__ == "__main__":
    run()