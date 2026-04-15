from pyspark.sql import DataFrame
from pyspark.sql.window import Window
from pyspark.sql.functions import lead


def build_target(df: DataFrame, horizon: int = 1) -> DataFrame:
    """
    Tworzy target (przyszłą wartość metryki)

    horizon = ile kroków do przodu przewidujemy
    """

    window_spec = Window.partitionBy("entity_id", "metric_type").orderBy("event_time")

    df = df.withColumn("target", lead("metric_value", horizon).over(window_spec))

    return df