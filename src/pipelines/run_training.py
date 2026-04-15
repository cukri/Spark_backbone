from utils.spark_session import get_spark
from models.train_model import train_model

def run_training_pipeline():
    spark = get_spark()

    train_df = spark.read.parquet("data/processed/train")
    test_df = spark.read.parquet("data/processed/test")

    model, predictions, rmse = train_model(train_df, test_df)

    print(f"RMSE: {rmse}")

    return model, predictions


def main():
    print("Running training pipeline...")
    model, predictions = run_training_pipeline()

    predictions.select("metric_value", "target", "prediction").show(10)


if __name__ == "__main__":
    main()