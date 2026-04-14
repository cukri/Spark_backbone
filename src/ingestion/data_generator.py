from datetime import datetime, timedelta
import random
import math

from pyspark.sql import DataFrame
from pyspark.sql.types import (
    StructType, StructField,
    TimestampType, StringType, DoubleType
)


# ------------------------
# Helper utilities
# ------------------------
def clip(value: float, min_val: float, max_val: float) -> float:
    return float(max(min_val, min(max_val, value)))


# ------------------------
# Metric generators
# ------------------------

def generate_cpu(day_index: int, drift_day: int) -> float:
    """
    CPU:
    - reactive
    - bounded [0, 100]
    - mean shift after drift_day
    """
    base_mean = 35
    drift_mean = 45
    std = 7

    mean = base_mean if day_index < drift_day else drift_mean
    value = random.gauss(mean, std)

    return clip(value, 0, 100)


def generate_memory(day_index: int) -> float:
    """
    Memory:
    - stable
    - slow upward trend (e.g. cache, leaks)
    - bounded [0, 100]
    """
    base_mean = 40
    trend_per_day = 0.02   # very slow drift
    std = 2

    mean = base_mean + day_index * trend_per_day
    value = random.gauss(mean, std)

    return clip(value, 0, 100)


def generate_response_time(day_index: int, drift_day: int) -> float:
    """
    Response time:
    - log-normal (heavy tails)
    - variance drift after drift_day
    - unbounded upper tail (clipped only for sanity)
    """
    base_mean_ms = 120
    std_pre = 0.35
    std_post = 0.55

    std = std_pre if day_index < drift_day else std_post

    # log-normal sampling
    value = random.lognormvariate(
        math.log(base_mean_ms),
        std
    )

    # soft sanity clipping (not physical bound)
    return clip(value, 20, 2000)


# ------------------------
# Main generator
# ------------------------

def generate_data(
    spark,
    start_date: str = "2024-01-01",
    days: int = 365,
    freq_minutes: int = 5,
    entities: int = 5,
    drift_day: int = 100,
    seed: int = 42
) -> DataFrame:
    """
    Generate realistic multi-metric time series data with:
    - controlled drift
    - latency
    - missing values
    """

    random.seed(seed)

    start_dt = datetime.fromisoformat(start_date)
    rows = []

    steps_per_day = int(24 * 60 / freq_minutes)
    total_steps = days * steps_per_day

    entity_ids = [f"service_{i}" for i in range(entities)]
    metric_types = ["cpu", "memory", "response_time"]

    for step in range(total_steps):
        event_time = start_dt + timedelta(minutes=step * freq_minutes)
        day_index = step // steps_per_day

        for entity in entity_ids:
            for metric_type in metric_types:

                if metric_type == "cpu":
                    value = generate_cpu(day_index, drift_day)

                elif metric_type == "memory":
                    value = generate_memory(day_index)

                elif metric_type == "response_time":
                    value = generate_response_time(day_index, drift_day)

                else:
                    continue

                # ------------------------
                # Missing data (2%)
                # ------------------------
                if random.random() < 0.02:
                    value = None

                # ------------------------
                # Latency: data available 0–48h later
                # ------------------------
                latency_hours = random.uniform(0, 48)
                available_at = event_time + timedelta(hours=latency_hours)

                rows.append(
                    (
                        event_time,
                        available_at,
                        entity,
                        metric_type,
                        value,
                    )
                )

    schema = StructType([
        StructField("event_time", TimestampType(), False),
        StructField("available_at", TimestampType(), False),
        StructField("entity_id", StringType(), False),
        StructField("metric_type", StringType(), False),
        StructField("metric_value", DoubleType(), True),
    ])

    return spark.createDataFrame(rows, schema)