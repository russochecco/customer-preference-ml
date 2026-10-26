import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark_session():
    """
    Initializes a fast, local Spark Session explicitly configured for
    isolated, in-memory automated test environments.
    """
    spark = (
        SparkSession.builder
        .master("local")  # Run locally with 2 worker threads
        .appName("CustomerPreferenceTesting")
        .config("spark.sql.shuffle.partitions", "1")  # Low partitions speed up tests
        .getOrCreate()
    )

    yield spark

    spark.stop()
