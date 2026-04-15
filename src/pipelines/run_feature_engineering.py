from utils.spark_session import get_spark
from features.feature_engineering import build_features


def main():
    spark = get_spark()

    print("Loading raw data...")
    df = spark.read.parquet("data/raw/metrics")

    print("Building features...")
    df = build_features(df)

    output_path = "data/processed/features"
    df.write.mode("overwrite").parquet(output_path)

    print(f"Features saved to {output_path}")


if __name__ == "__main__":
    main()