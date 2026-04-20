from utils.spark_session import get_spark
from pyspark.sql.functions import col, count, when, mean, stddev


def run():
    spark = get_spark()

    spark.sparkContext.setLogLevel("ERROR")

    print("\n=== DATA QUALITY ANALYSIS ===\n", flush=True)

    # ======================
    # LOAD RAW
    # ======================
    print("Loading raw data...", flush=True)
    df = spark.read.parquet("data/raw/metrics").cache()

    # ======================
    # BASIC INFO
    # ======================
    print("\n=== BASIC INFO ===")

    row_count = df.count()
    print(f"Row count: {row_count:,}")  # ładne formatowanie

    print("\nColumns:")
    print(", ".join(df.columns))

    # ======================
    # NULLS
    # ======================
    print("\n=== NULL ANALYSIS ===")

    null_counts = df.select([
        count(when(col(c).isNull(), c)).alias(c)
        for c in df.columns
    ])

    null_counts.show(truncate=False)

    # ======================
    # TARGET DISTRIBUTION
    # ======================
    print("\n=== METRIC VALUE STATS ===")

    df.select("metric_value").describe().show()

    df.selectExpr(
        "min(metric_value)",
        "max(metric_value)",
        "avg(metric_value)",
        "stddev(metric_value)"
    ).show()

    # ======================
    # QUANTILES
    # ======================
    print("\n=== QUANTILES ===")

    quantiles = df.approxQuantile(
        "metric_value",
        [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
        0.01
    )

    labels = ["1%", "5%", "25%", "50%", "75%", "95%", "99%"]
    for l, q in zip(labels, quantiles):
        print(f"{l:>4} : {q:.4f}")

    # ======================
    # OUTLIERS
    # ======================
    print("\n=== OUTLIERS (STD) ===")

    stats = df.select(
        mean("metric_value").alias("mean"),
        stddev("metric_value").alias("std")
    ).collect()[0]

    mean_val = stats["mean"]
    std_val = stats["std"]

    print(f"Mean: {mean_val:.4f}")
    print(f"Std : {std_val:.4f}")

    lower_2std = mean_val - 2 * std_val
    upper_2std = mean_val + 2 * std_val

    print(f"Lower bound: {lower_2std:.4f}")
    print(f"Upper bound: {upper_2std:.4f}")

    outliers = df.filter(
        (col("metric_value") < lower_2std) |
        (col("metric_value") > upper_2std)
    ).count()

    print(f"Outliers (>2 std): {outliers:,}")
    print(f"% of dataset: {outliers / row_count * 100:.2f}%")

    # ======================
    # DUPLICATES
    # ======================
    print("\n=== DUPLICATES ===")

    dedup_count = df.dropDuplicates().count()
    dup_count = row_count - dedup_count

    print(f"Duplicate rows: {dup_count:,}")

    print("\n=== END ===\n")


if __name__ == "__main__":
    run()