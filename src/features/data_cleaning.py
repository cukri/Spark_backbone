from pyspark.sql.functions import col, when, log1p


def clean_metrics(df, upper_quantile=0.95):
    """
    Cleans raw metrics data:
    - removes nulls
    - clips outliers
    - applies log transform

    Parameters:
        df (DataFrame): raw dataframe
        upper_quantile (float): quantile for clipping

    Returns:
        DataFrame: cleaned dataframe
    """

    # ======================
    # DROP NULLS
    # ======================
    df = df.filter(col("metric_value").isNotNull())

    # ======================
    # COMPUTE CLIP VALUE
    # ======================
    upper_cap = df.approxQuantile("metric_value", [upper_quantile], 0.01)[0]

    # ======================
    # CLIPPING
    # ======================
    df = df.withColumn(
        "metric_value_clipped",
        when(col("metric_value") > upper_cap, upper_cap)
        .otherwise(col("metric_value"))
    )

    # ======================
    # LOG TRANSFORM
    # ======================
    df = df.withColumn(
        "metric_value_log",
        log1p(col("metric_value_clipped"))
    )

    return df