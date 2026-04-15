from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator


def train_model(train_df: DataFrame, test_df: DataFrame):
    """
    Trenuje model regresyjny i ocenia go
    """

    feature_cols = [
        "lag_1",
        "delta_1",
        "rolling_avg_3",
        "hour",
        "day_of_week"
    ]

    # vector
    assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="skip"
    )

    train_df = assembler.transform(train_df)
    test_df = assembler.transform(test_df)

    # model
    model = LinearRegression(
        featuresCol="features",
        labelCol="target"
    )

    #  trening
    model = model.fit(train_df)

    #  predykcja
    predictions = model.transform(test_df)

    #  ewaluacja
    evaluator = RegressionEvaluator(
        labelCol="target",
        predictionCol="prediction",
        metricName="rmse"
    )

    rmse = evaluator.evaluate(predictions)

    print(f"RMSE: {rmse}")

    return model, predictions, rmse