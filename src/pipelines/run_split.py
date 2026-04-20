from utils.spark_session import get_spark
from features.data_splitter import time_based_split


def run():
    spark = get_spark()

    print("Loading metrics...")
    df = spark.read.parquet("data/processed/features")

    print("Splitting...")
    train_df, test_df = time_based_split(df)

    print("Train count:", train_df.count())
    print("Test count:", test_df.count())

    train_df.write.mode("overwrite").parquet("data/processed/train")
    test_df.write.mode("overwrite").parquet("data/processed/test")

    print("Train/Test saved")


if __name__ == "__main__":
    run()