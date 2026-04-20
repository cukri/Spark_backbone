from pyspark.sql import SparkSession


def get_spark(app_name="spark-ml-backbone"):
    return (
        SparkSession.builder
        .appName(app_name)

        # mniej równoległości
        .master("local[4]")

        # RAM
        .config("spark.driver.memory", "6g")
        .config("spark.executor.memory", "6g")

        # mniej shuffle
        .config("spark.sql.shuffle.partitions", "4")

        # KLUCZOWE
        .config("spark.sql.parquet.enableDictionary", "false")

        # mniejsze row groups parquet
        .config("parquet.block.size", "33554432")   # 32MB

        .config("spark.python.worker.reuse", "true")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")
    return spark