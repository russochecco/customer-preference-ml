import os
import sys
import logging
from pyspark.sql import DataFrame, SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import explode

APP_NAME = "ImplicitPipeline"
log = logging.getLogger(APP_NAME)


def setup_logging():
    """Configures systemic pipeline logging diagnostics."""
    if not log.hasHandlers():
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt='%(asctime)s %(levelname)s %(name)s: %(message)s',
            datefmt="%y/%m/%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        log.addHandler(console_handler)
    log.setLevel(logging.INFO)


def get_spark_session() -> SparkSession:
    """Creates or retrieves a standard execution context Spark Session."""
    return (
        SparkSession.builder
        .appName(APP_NAME)
        .config("spark.sql.pyspark.worker.logging.enabled", "true")
        .getOrCreate()
    )


def transform_recommendations(user_recommendations: DataFrame, users_df: DataFrame,
                              products_df: DataFrame) -> DataFrame:
    """Transforms raw recommendation array structures to relational rows with metadata."""
    exploded_recommendations = (
        user_recommendations.withColumn("rec", explode("recommendations"))
        .select("user_id", "rec.product_id", "rec.rating")
    )
    return (
        exploded_recommendations.join(users_df, on="user_id", how="inner")
        .join(products_df, on="product_id", how="inner")
        .select("name", "signup_country", "product_name", "category", "rating")
        .withColumnRenamed("rating", "confidence_score")
        .orderBy("name")
    )


def run_pipeline(spark: SparkSession, config: dict):
    """Executes the complete core implicit recommendations matrix pipeline training loop."""
    log.info("Initializing Implicit Preferences ML Pipeline...")

    # Data ingestion happens naturally; failures bubble up automatically
    log.info("Ingesting datasets...")
    users_df = spark.read.csv(config["USERS_DATASET_PATH"], header=True, inferSchema=True)
    products_df = spark.read.csv(config["PRODUCTS_DATASET_PATH"], header=True, inferSchema=True)
    implicit_prefs_df = spark.read.csv(config["IMPLICIT_PREFERENCES_DATASET_PATH"], header=True, inferSchema=True)

    log.info("Configuring ALS estimator for implicit confidence mapping...")
    als_implicit = ALS(
        maxIter=15,
        regParam=0.01,
        userCol="user_id",
        itemCol="product_id",
        ratingCol="interaction_count",
        implicitPrefs=True,
        alpha=float(config["ALPHA"]),
        coldStartStrategy="drop"
    )

    log.info("Training implicit interaction confidence model...")
    model_implicit = als_implicit.fit(implicit_prefs_df)

    log.info(f"Generating top {config['NUM_ITEMS_PER_USER']} implicit recommendations...")
    user_recommendations = model_implicit.recommendForAllUsers(int(config["NUM_ITEMS_PER_USER"]))

    recommendations_matrix = transform_recommendations(user_recommendations, users_df, products_df)

    log.info("Displaying final recommendation matrix output:")
    recommendations_matrix.show(truncate=False)

    # Model saving happens naturally
    log.info(f"Saving serialized model pipeline weights to disk: {config['MODEL_PATH']}")
    model_implicit.write().overwrite().save(config["MODEL_PATH"])


if __name__ == "__main__":
    setup_logging()

    # Pack env requirements safely into a dictionary ONLY when executed directly
    pipeline_config = {
        "MODEL_PATH": os.environ["MODEL_PATH"],
        "USERS_DATASET_PATH": os.environ["USERS_DATASET_PATH"],
        "PRODUCTS_DATASET_PATH": os.environ["PRODUCTS_DATASET_PATH"],
        "IMPLICIT_PREFERENCES_DATASET_PATH": os.environ["IMPLICIT_PREFERENCES_DATASET_PATH"],
        "ALPHA": os.environ["ALPHA"],
        "NUM_ITEMS_PER_USER": os.environ["NUM_ITEMS_PER_USER"],
    }

    session = get_spark_session()

    try:
        run_pipeline(session, pipeline_config)
    except Exception as e:
        log.exception("Pipeline execution failed contextually:")
        sys.exit(1)
    finally:
        log.info("Stopping Spark Application Session Context safely.")
        session.stop()
