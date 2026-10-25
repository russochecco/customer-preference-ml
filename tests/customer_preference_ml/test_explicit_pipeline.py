import os
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType, FloatType
from customer_preference_ml.explicit_pipeline import transform_recommendations, run_pipeline


@pytest.fixture(scope="session")
def spark_session():
    """
    Initializes a fast, local Spark Session explicitly configured for
    isolated, in-memory automated test environments.
    """
    spark = (
        SparkSession.builder
        .master("local")  # Run locally with 2 worker threads
        .appName("ExplicitPipelineTesting")
        .config("spark.sql.shuffle.partitions", "1")  # Low partitions speed up tests
        .getOrCreate()
    )
    yield spark
    spark.stop()


@pytest.fixture
def mock_pipeline_config(tmp_path):
    """
    Generates a localized configuration dictionary mapping to an isolated
    temporary directory provided by pytest to avoid cluttering local file structures.
    """
    base_dir = tmp_path / "data"
    base_dir.mkdir()

    return {
        "MODEL_PATH": str(tmp_path / "models" / "explicit_als_model"),
        "USERS_DATASET_PATH": str(base_dir / "users.csv"),
        "PRODUCTS_DATASET_PATH": str(base_dir / "products.csv"),
        "EXPLICIT_PREFERENCES_DATASET_PATH": str(base_dir / "explicit_preferences.csv"),
        "NUM_ITEMS_PER_USER": "2"
    }


# ============================================================================
# SECTION 1: UNIT TESTING ISOLATED DATA TRANSFORMATIONS
# ============================================================================

def test_transform_recommendations_maps_metadata_successfully(spark_session):
    """
    Validates that transform_recommendations correctly explodes the recommendation
    struct array and links corresponding product, user, and membership tier metadata.
    """
    # 1. Arrange: Initialize DataFrames with precise testing schemas
    user_rec_schema = StructType([
        StructField("user_id", IntegerType(), False),
        StructField("recommendations", ArrayType(
            StructType([
                StructField("product_id", IntegerType(), False),
                StructField("rating", FloatType(), False)
            ])
        ), False)
    ])

    users_df = spark_session.createDataFrame(
        [(1, "Bob", "Gold")],
        ["user_id", "name", "membership_tier"]
    )
    products_df = spark_session.createDataFrame(
        [(101, "Wireless Mouse", "Electronics")],
        ["product_id", "product_name", "category"]
    )

    # Simulate the raw output dataset structure natively returned by ALS model
    raw_recommendations_df = spark_session.createDataFrame(
        [(1, [(101, 4.5)])],
        schema=user_rec_schema
    )

    # 2. Act: Execute the business transformation function
    result_df = transform_recommendations(raw_recommendations_df, users_df, products_df)
    extracted_rows = result_df.collect()

    # 3. Assert: Validate schemas, outputs, and metadata mapping properties
    assert len(extracted_rows) == 1
    assert "membership_tier" in result_df.columns

    transformed_record = extracted_rows[0]
    assert transformed_record["name"] == "Bob"
    assert transformed_record["membership_tier"] == "Gold"
    assert transformed_record["product_name"] == "Wireless Mouse"
    assert transformed_record["rating"] == 4.5


def test_transform_recommendations_drops_unmatched_records(spark_session):
    """
    Ensures inner join mechanics drop recommendation entities if they cannot be
    reconciled against product or user registration metadata catalogues.
    """
    user_rec_df = spark_session.createDataFrame(
        [(99, [(777, 5.0)])],
        ["user_id", "recommendations"]
    ).withColumn("recommendations", F.expr("cast(recommendations as array<struct<product_id:int,rating:float>>)"))

    # Bob is User #1, but the recommendation dataset belongs to unlisted User #99
    users_df = spark_session.createDataFrame([(1, "Bob", "Gold")], ["user_id", "name", "membership_tier"])
    products_df = spark_session.createDataFrame([(777, "Wireless Mouse", "Electronics")],
                                                ["product_id", "product_name", "category"])

    result_df = transform_recommendations(user_rec_df, users_df, products_df)

    assert result_df.count() == 0


# ============================================================================
# SECTION 2: INTEGRATION TESTING CORE RUN LOOPS
# ============================================================================

def test_run_pipeline_handles_missing_file_exception(spark_session, mock_pipeline_config):
    """
    Ensures that calling run_pipeline when physical data dependencies do not
    exist outputs a descriptive exception that allows clean process exit.
    """
    # Act & Assert: run_pipeline should bubble up ingestion failure cleanly
    with pytest.raises(Exception):
        run_pipeline(spark_session, mock_pipeline_config)


def test_run_pipeline_full_execution_cycle(spark_session, mock_pipeline_config):
    """
    End-to-End Integration Check: Feeds real physical mocked data files through
    the system to verify ingestion, training, recommendation, and writing.
    """
    # 1. Arrange: Write mock CSV assets to the dynamic temp locations
    spark_session.createDataFrame(
        [(1, "Alice", "Gold"), (2, "Bob", "Silver")],
        ["user_id", "name", "membership_tier"]
    ).write.csv(mock_pipeline_config["USERS_DATASET_PATH"], header=True, mode="overwrite")

    spark_session.createDataFrame(
        [(101, "Laptop", "Tech"), (102, "Coffee", "Food")],
        ["product_id", "product_name", "category"]
    ).write.csv(mock_pipeline_config["PRODUCTS_DATASET_PATH"], header=True, mode="overwrite")

    # In explicit systems, ratings are explicit user-given scores (e.g., 1-5 stars)
    spark_session.createDataFrame(
        [(1, 101, 5.0), (1, 102, 2.0), (2, 102, 4.0), (2, 101, 1.0)],
        ["user_id", "product_id", "rating"]
    ).write.csv(mock_pipeline_config["EXPLICIT_PREFERENCES_DATASET_PATH"], header=True, mode="overwrite")

    # 2. Act: Fire the pipeline execution context
    run_pipeline(spark_session, mock_pipeline_config)

    # 3. Assert: Confirm model artifact serialization folder was generated on disk
    assert os.path.exists(mock_pipeline_config["MODEL_PATH"])
    assert os.path.isdir(mock_pipeline_config["MODEL_PATH"])
