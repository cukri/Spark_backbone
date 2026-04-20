from pyspark.sql import DataFrame
from pyspark.sql.window import Window
from pyspark.sql.functions import (
    col,
    lag,
    avg,
    stddev,
    min as spark_min,
    max as spark_max,
    hour,
    dayofweek,
    month,
    sin,
    cos,
    lit
)


def build_features(df: DataFrame) -> DataFrame:
    """
    Advanced Feature Engineering v3

    Features:
    - multiple lags
    - momentum / deltas
    - rolling mean/std/min/max
    - ratio to averages
    - volatility
    - cyclical time features
    - future target
    """

    # ==================================================
    # USE CLEAN TARGET
    # ==================================================
    value_col = "metric_value_log"

    # ==================================================
    # WINDOW
    # ==================================================
    window_spec = Window.partitionBy(
        "entity_id", "metric_type"
    ).orderBy("event_time")

    # ==================================================
    # LAGS
    # ==================================================
    lag_list = [1, 2, 3, 6, 12, 24]

    for l in lag_list:
        df = df.withColumn(f"lag_{l}", lag(value_col, l).over(window_spec))

    # ==================================================
    # MOMENTUM / DELTAS
    # ==================================================
    df = df.withColumn("delta_1", col(value_col) - col("lag_1"))
    df = df.withColumn("delta_3", col(value_col) - col("lag_3"))
    df = df.withColumn("delta_6", col(value_col) - col("lag_6"))
    df = df.withColumn("delta_12", col(value_col) - col("lag_12"))

    df = df.withColumn("pct_change_1", col("delta_1") / (col("lag_1") + lit(0.0001)))
    df = df.withColumn("pct_change_3", col("delta_3") / (col("lag_3") + lit(0.0001)))

    # ==================================================
    # ROLLING WINDOWS
    # ==================================================
    rolling_3 = window_spec.rowsBetween(-3, -1)
    rolling_6 = window_spec.rowsBetween(-6, -1)
    rolling_12 = window_spec.rowsBetween(-12, -1)
    rolling_24 = window_spec.rowsBetween(-24, -1)

    windows = {
        3: rolling_3,
        6: rolling_6,
        12: rolling_12,
        24: rolling_24
    }

    for n, w in windows.items():
        df = df.withColumn(f"rolling_avg_{n}", avg(value_col).over(w))
        df = df.withColumn(f"rolling_std_{n}", stddev(value_col).over(w))
        df = df.withColumn(f"rolling_min_{n}", spark_min(value_col).over(w))
        df = df.withColumn(f"rolling_max_{n}", spark_max(value_col).over(w))

    # ==================================================
    # POSITION VS HISTORY
    # ==================================================
    df = df.withColumn(
        "ratio_to_avg_6",
        col(value_col) / (col("rolling_avg_6") + lit(0.0001))
    )

    df = df.withColumn(
        "ratio_to_avg_24",
        col(value_col) / (col("rolling_avg_24") + lit(0.0001))
    )

    # ==================================================
    # RANGE FEATURES
    # ==================================================
    df = df.withColumn(
        "range_6",
        col("rolling_max_6") - col("rolling_min_6")
    )

    df = df.withColumn(
        "range_24",
        col("rolling_max_24") - col("rolling_min_24")
    )

    # ==================================================
    # TIME FEATURES
    # ==================================================
    df = df.withColumn("hour", hour("event_time"))
    df = df.withColumn("day_of_week", dayofweek("event_time"))
    df = df.withColumn("month", month("event_time"))

    # cyclical hour encoding
    df = df.withColumn("hour_sin", sin(col("hour") * 2 * 3.14159 / 24))
    df = df.withColumn("hour_cos", cos(col("hour") * 2 * 3.14159 / 24))

    # ==================================================
    # TARGET (NEXT STEP)
    # ==================================================
    df = df.withColumn(
        "target",
        lag(value_col, -1).over(window_spec)
    )

    # ==================================================
    # CLEAN NULLS
    # ==================================================
    df = df.dropna()

    return df