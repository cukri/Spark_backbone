from pipelines.run_ingestion import run as run_ingestion
from pipelines.run_feature_engineering import run as run_features
from pipelines.run_split import run as run_split


def main():
    run_ingestion()
    run_features()
    run_split()


if __name__ == "__main__":
    main()