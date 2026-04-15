from pyspark.sql import DataFrame
from pyspark.sql.functions import col


def time_based_split(df: DataFrame, split_ratio: float = 0.8):
    """
    Dzieli dane na train/test na podstawie czasu (bez leakage)
    """

    # znajdź min i max czasu
    min_time = df.selectExpr("min(event_time)").collect()[0][0]
    max_time = df.selectExpr("max(event_time)").collect()[0][0]

    # punkt podziału
    split_point = min_time + (max_time - min_time) * split_ratio

    # split
    train_df = df.filter(col("event_time") <= split_point)
    test_df = df.filter(col("event_time") > split_point)

    return train_df, test_df