from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator


def train_model(train_df, test_df, feature_cols):

    # ======================
    # DROP NULLS
    # ======================
    train_df = train_df.dropna(subset=feature_cols + ["target"])
    test_df = test_df.dropna(subset=feature_cols + ["target"])

    # ======================
    # VECTOR ASSEMBLER
    # ======================
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )

    train_df = assembler.transform(train_df)
    test_df = assembler.transform(test_df)

    # ======================
    # GBT REGRESSOR
    # ======================
    gbt = GBTRegressor(
        labelCol="target",
        featuresCol="features",

        maxIter=100,       # liczba drzew
        maxDepth=6,        # głębokość drzewa
        stepSize=0.05,     # learning rate
        subsamplingRate=0.8,

        seed=42
    )

    model = gbt.fit(train_df)

    # ======================
    # PREDICTIONS
    # ======================
    predictions = model.transform(test_df)

    # ======================
    # RMSE
    # ======================
    evaluator = RegressionEvaluator(
        labelCol="target",
        predictionCol="prediction",
        metricName="rmse"
    )

    rmse = evaluator.evaluate(predictions)

    return model, predictions, rmse