from xml.parsers.expat import model

from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator


def train_model(train_df, test_df, feature_cols):

    train_df = train_df.dropna(subset=feature_cols + ["target"])
    test_df = test_df.dropna(subset=feature_cols + ["target"])

    # assembler
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )

    train_df = assembler.transform(train_df)
    test_df = assembler.transform(test_df)

    # model
    lr = LinearRegression(labelCol="target", featuresCol="features")
    model = lr.fit(train_df)

    predictions = model.transform(test_df)

    # RMSE
    evaluator = RegressionEvaluator(
        labelCol="target",
        predictionCol="prediction",
        metricName="rmse"
    )

    rmse = evaluator.evaluate(predictions)

    return model, predictions, rmse