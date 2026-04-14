from utils.spark_session import get_spark
from ingestion.data_generator import generate_data


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

    # zapis
    output_path = "data/raw/metrics"

    (
        df.limit(1000)
        .write
        .mode("overwrite")
        .parquet(output_path)
    )

    print(f"Data saved to {output_path}")

    df.show(5)
    df.printSchema()


if __name__ == "__main__":
    main()