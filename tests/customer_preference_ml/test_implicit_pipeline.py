import pytest
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType, FloatType
from customer_preference_ml.implicit_pipeline import transform_recommendations, run_pipeline


@pytest.fixture(scope="session")
def spark_session():
    """
    Initializes a fast, local Spark Session explicitly configured for
    isolated, in-memory automated test environments.
    """
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("ImplicitPipelineTesting")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )

    yield spark
    spark.stop()


@pytest.fixture
def mock_pipeline_config(tmp_path):
    """
    Generates a localized configuration dictionary mapping to a real, isolated
    temporary directory provided by pytest to avoid touching actual file environments.
    """
    base_dir = tmp_path / "data"
    base_dir.mkdir()

    return {
        "MODEL_PATH": str(tmp_path / "models" / "implicit_model"),
        "USERS_DATASET_PATH": str(base_dir / "users.csv"),
        "PRODUCTS_DATASET_PATH": str(base_dir / "products.csv"),
        "IMPLICIT_PREFERENCES_DATASET_PATH": str(base_dir / "preferences.csv"),
        "ALPHA": "40.0",
        "NUM_ITEMS_PER_USER": "2"
    }


# ============================================================================
# SECTION 1: UNIT TESTING ISOLATED DATA TRANSFORMATIONS
# ============================================================================

def test_transform_recommendations_maps_metadata_successfully(spark_session):
    """
    Validates that transform_recommendations correctly explodes the recommendation
    struct array and links corresponding product and user metadata together.
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
        [(10, "Alice", "IT")],
        ["user_id", "name", "signup_country"]
    )
    products_df = spark_session.createDataFrame(
        [(500, "Espresso Beans", "Coffee"), (600, "Moka Pot", "Kitchen")],
        ["product_id", "product_name", "category"]
    )

    # Simulate the raw output dataset structure natively returned by ALS model
    raw_recommendations_df = spark_session.createDataFrame(
        [(10, [(500, 3.8), (600, 1.2)])],
        schema=user_rec_schema
    )

    # 2. Act: Execute the business transformation function
    result_df = transform_recommendations(raw_recommendations_df, users_df, products_df)
    extracted_rows = result_df.collect()

    # 3. Assert: Validate schemas, execution outputs, and ordering requirements
    assert len(extracted_rows) == 2
    assert "confidence_score" in result_df.columns
    assert "rating" not in result_df.columns  # Column should have been renamed

    first_transformed_record = extracted_rows[0]
    assert first_transformed_record["name"] == "Alice"
    assert first_transformed_record["signup_country"] == "IT"
    assert first_transformed_record["product_name"] in ["Espresso Beans", "Moka Pot"]


def test_transform_recommendations_drops_unmatched_records(spark_session):
    """
    Ensures inner join mechanics drop recommendation entities if they cannot be
    reconciled against product or user registration metadata catalogues.
    """
    user_rec_df = spark_session.createDataFrame(
        [(99, [(888, 4.0)])],
        ["user_id", "recommendations"]
    ).withColumn("recommendations", F.expr("cast(recommendations as array<struct<product_id:int,rating:float>>)"))

    # Alice is User #1, but the recommendation belongs to unlisted User #99
    users_df = spark_session.createDataFrame([(1, "Alice", "IT")], ["user_id", "name", "signup_country"])
    products_df = spark_session.createDataFrame([(888, "Espresso Beans", "Coffee")],
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
    # Act & Assert: run_pipeline should bubbles up ingestion failure cleanly
    with pytest.raises(Exception):
        run_pipeline(spark_session, mock_pipeline_config)


def test_run_pipeline_full_execution_cycle(spark_session, mock_pipeline_config):
    """
    End-to-End Integration Check: Feeds real physical mocked data files through
    the system to verify ingestion, training, recommendation, and writing.
    """
    # 1. Arrange: Write mock CSV assets to the dynamic temp locations
    spark_session.createDataFrame(
        [(1, "Alice", "IT"), (2, "Bob", "US")],
        ["user_id", "name", "signup_country"]
    ).write.csv(mock_pipeline_config["USERS_DATASET_PATH"], header=True, mode="overwrite")

    spark_session.createDataFrame(
        [(101, "Laptop", "Tech"), (102, "Coffee", "Food")],
        ["product_id", "product_name", "category"]
    ).write.csv(mock_pipeline_config["PRODUCTS_DATASET_PATH"], header=True, mode="overwrite")

    # In implicit systems, ratings represent interaction frequency counts
    spark_session.createDataFrame(
        [(1, 101, 5), (1, 102, 1), (2, 102, 12)],
        ["user_id", "product_id", "interaction_count"]
    ).write.csv(mock_pipeline_config["IMPLICIT_PREFERENCES_DATASET_PATH"], header=True, mode="overwrite")

    # 2. Act: Fire the pipeline execution context
    run_pipeline(spark_session, mock_pipeline_config)

    # 3. Assert: Confirm model artifact serialization folder was generated on disk
    import os
    assert os.path.exists(mock_pipeline_config["MODEL_PATH"])
    assert os.path.isdir(mock_pipeline_config["MODEL_PATH"])
