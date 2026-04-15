from utils.spark_session import get_spark
from ingestion.data_generator import generate_data
from features.feature_engineering import build_features
from features.data_splitter import time_based_split
from features.target_builder import build_target


def main():
    spark = get_spark()

    print("Generating data...")

    df = generate_data(
        spark=spark,
        start_date="2024-01-01",
        days=365,
        entities=5,
        drift_day=100
    )

    print(f"Rows generated: {df.count()}")

    # 🔹 SAVE RAW
    raw_path = "data/raw/metrics"
    df.write.mode("overwrite").parquet(raw_path)
    print(f"Raw data saved to {raw_path}")

    # 🔹 FEATURE ENGINEERING
    print("Building features...")
    df = build_features(df)

    # 🔹 TARGET
    print("Building target...")
    df = build_target(df)

    # 🔹 CLEAN
    df = df.dropna(subset=["target"])

    # 🔹 SPLIT
    train_df, test_df = time_based_split(df)

    print(f"TRAIN: {train_df.count()}")
    print(f"TEST: {test_df.count()}")

    # 🔹 SAVE PROCESSED
    processed_path = "data/processed/metrics"
    df.write.mode("overwrite").parquet(processed_path)

    print(f"Processed data saved to {processed_path}")

    # DEBUG
    df.show(5)
    df.printSchema()


if __name__ == "__main__":
    main()