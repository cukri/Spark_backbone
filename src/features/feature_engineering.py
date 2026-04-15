from pyspark.sql import DataFrame
from pyspark.sql.window import Window
from pyspark.sql.functions import (
    col, lag, avg, hour, dayofweek
)


def build_features(df: DataFrame) -> DataFrame:
    """
    Tworzy feature'y pod model ML
    """

    # okno czasowe per entity + metric
    window_spec = Window.partitionBy("entity_id", "metric_type").orderBy("event_time")

    # LAG (poprzednia wartość)
    df = df.withColumn("lag_1", lag("metric_value", 1).over(window_spec))

    # DELTA (zmiana)
    df = df.withColumn("delta_1", col("metric_value") - col("lag_1"))

    # ROLLING AVG (trend)
    rolling_window = window_spec.rowsBetween(-3, 0)
    df = df.withColumn("rolling_avg_3", avg("metric_value").over(rolling_window))

    # TIME FEATURES
    df = df.withColumn("hour", hour("event_time"))
    df = df.withColumn("day_of_week", dayofweek("event_time"))

    return df