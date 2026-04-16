from utils.spark_session import get_spark
from models.train_model import train_model


def run_training_pipeline(feature_cols):
    spark = get_spark()

    print("Loading train/test...", flush=True)

    train_df = spark.read.parquet("data/processed/train")
    test_df = spark.read.parquet("data/processed/test")

    print("Training model...", flush=True)

    model, predictions, rmse = train_model(train_df, test_df, feature_cols)

    print("Training finished", flush=True)

    return model, predictions, rmse


def main():

    # coefficients
    feature_cols = [
        "lag_1",
        "delta_1",
        "rolling_avg_3",
        "hour",
        "day_of_week"
    ]

    model, predictions, rmse = run_training_pipeline(feature_cols)

    print("\nModel coefficients:", flush=True)
    for name, coef in zip(feature_cols, model.coefficients):
        print(f"{name}: {coef}", flush=True)

    print(f"Intercept: {model.intercept}", flush=True)

    print("\n=== SHOWING PREDICTIONS ===", flush=True)

    predictions.select("metric_value", "target", "prediction").show(10)

    print("=== END ===", flush=True)


if __name__ == "__main__":
    main()