from utils.spark_session import get_spark


def run():
    print("=== RUN_ANALYSIS STARTED ===", flush=True)

    spark = get_spark()

    # ======================
    # LOAD DATA
    # ======================
    print("Loading train data...", flush=True)
    train_df = spark.read.parquet("data/processed/train")
    print("Data loaded", flush=True)

    # ======================
    # BASIC INFO
    # ======================
    print("\n=== BASIC INFO ===", flush=True)

    print("Counting rows...", flush=True)
    row_count = train_df.count()
    print(f"Row count: {row_count}", flush=True)

    # ======================
    # TARGET ANALYSIS
    # ======================
    print("\n=== TARGET STATS ===", flush=True)

    train_df.select("target").describe().show()

    train_df.selectExpr(
        "min(target)",
        "max(target)",
        "avg(target)",
        "stddev(target)"
    ).show()

    # ======================
    # FEATURE CORRELATION
    # ======================
    print("\n=== CORRELATIONS WITH TARGET ===\n", flush=True)

    feature_cols = [
        "lag_1",
        "delta_1",
        "rolling_avg_3",
        "hour",
        "day_of_week"
    ]

    for col in feature_cols:
        corr = train_df.stat.corr(col, "target")
        print(f"{col} vs target: {corr}", flush=True)

    print("\n=== ANALYSIS DONE ===", flush=True)


if __name__ == "__main__":
    run()