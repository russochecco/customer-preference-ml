# Customer Preference ML Pipeline

A scalable, production-grade PySpark Machine Learning project that leverages Alternating Least Squares (ALS) matrix factorization to deliver customer product recommendations. The engine implements two distinct algorithmic training paradigms – **Implicit Interaction Confidence Mapping** and **Explicit Feedback Factorization** – built entirely on **PySpark 4.2.0** using a highly decoupled, maintainable, and fully testable architecture.

## Project Structure

The project follows a standard Python project layout with a clean separation between source code, datasets, automated tests, and environment configurations.

```text
.
├── README.md
├── data/
│   ├── explicit_preferences.csv     # Structured explicit review datasets (ratings)
│   ├── implicit_preferences.csv     # Raw behavior event streams (interaction counts)
│   ├── products.csv                 # Product item catalogue metadata registry
│   └── users.csv                    # Customer profiles and membership tiers
├── models/                          # Local target directory for serialized ML model weights
├── pyproject.toml                   # Project dependencies and tool configurations (Poetry)
├── src/
│   ├── explicit_pipeline.py         # Explicit rating matrix execution script
│   └── implicit_pipeline.py         # Implicit engagement confidence tracking script
└── tests/                           # Unit test suites mirroring source code structures
    ├── conftest.py                  # Global Pytest fixtures (shared SparkSession management)
    ├── test_explicit_pipeline.py
    └── test_implicit_pipeline.py
```

## Architectural Overview & Theory

The framework isolates recommendations based on the core characteristic of data captured from customers:

| Feature / Metric | Implicit Pipeline (`implicit_pipeline.py`) | Explicit Pipeline (`explicit_pipeline.py`) |
| :--- | :--- | :--- |
| **Data Characteristic** | Passive consumer signals (e.g., clicks, total views, purchase volumes). | Direct, quantifiable user declarations (e.g., star ratings, reviews). |
| **Target Variable Column** | `interaction_count` (Unbound non-negative values representing intensity). | `rating` (Bound linear scoring ranges). |
| **ALS Configuration** | `implicitPrefs=True` | `implicitPrefs=False` |
| **Key Hyperparameters** | `maxIter=15`, `regParam=0.01`, $\alpha$ | `maxIter=15`, `regParam=0.1` |
| **Evaluation Strategy** | Validated over continuous preference spectrums; relies on tuning hyperparameter stability constraints. | Validated via `RegressionEvaluator` metrics (**Root Mean Squared Error - RMSE**) against an un-trained test partition split. |
| **Output Report Schema** | `name`, `signup_country`, `product_name`, `category`, `confidence_score` | `name`, `membership_tier`, `product_name`, `category`, `rating` |

### 1. Implicit Pipeline Architecture
Configures the Spark ALS estimator with `implicitPrefs=True`. Rather than fitting specific rating integers, it treats tracking metrics as interaction strength values. The calculations are adjusted by a dynamic **alpha ($\alpha$) modifier parameter**, mapping action frequencies to mathematical confidence variables prior to model weight persistence.

### 2. Explicit Pipeline Architecture
Configures the Spark ALS estimator with `implicitPrefs=False`. It divides user records using an 80/20 random split window to build out and validate explicit predictions on a test partition. Rows where user metadata matches product parameters are dynamically compiled into an actionable relational report including properties like `membership_tier`.

## Engineering & Production Highlights

Both modules align with enterprise production-grade design principles:
* **PySpark 4.2.0 Compatibility:** Fully optimized to run on the PySpark 4.2.0 ecosystem, utilizing its enhanced performance configurations, native memory management, and typing utilities.
* **Zero Global Side Effects on Import:** Scripts are completely stripped of direct, un-guarded `os.environ` dependency fetches. Environment variables are safely contained inside isolated dictionary validation blocks inside standard `if __name__ == "__main__":` entry points.
* **Functional Decoupling:** Procedural code is cleanly split into modular, type-hinted functions (`setup_logging`, `get_spark_session`, `transform_recommendations`, and `run_pipeline`).
* **Test Isolation Ready:** Rather than spinning up a hardcoded local runtime, the primary engine runner (`run_pipeline`) accepts a pre-configured `SparkSession` context and configuration parameter dictionary. This allows automation testing suites to inject lightweight memory allocations and temporary test files instantly.

## Getting Started

### 1. Environment Prerequisite Setup
Ensure you have Python 3.12+, Java JDK 17 (or compatible version for Spark 4.2.0), and Spark prerequisites configured locally. Initialize dependencies using Poetry:

<pre><code># Install framework requirements, development dependencies, and pytest utilities
poetry install
</pre></code>

### 2. Execution Run Paths via Spark Standalone Clusters

The execution examples below are configured specifically to submit jobs to a **Spark Standalone Cluster** setup matching the PySpark 4.2.0 ecosystem. Before submitting, ensure that your standalone master worker process is running and accessible at the specified URI (`spark://127.0.0.1:7077`).

> **Local Development Alternative:** If you do not have a standalone cluster manager running locally and want to run the scripts entirely within a local multi-threaded development environment, simply swap the `--master spark://127.0.0.1:7077` flag with `--master local[*]` in the shell scripts below.

#### Executing the Implicit ML Pipeline on a Standalone Cluster
To run the implicit configuration track, activate your environment container, pass your pipeline tuning configurations as shell environment variables, and route your job run directly via `spark-submit`:

<pre><code>#!/bin/bash

source .venv/bin/activate

# Configure pipeline environment parameters
export MODEL_PATH=models/implicit_als_model
export USERS_DATASET_PATH=data/users.csv
export PRODUCTS_DATASET_PATH=data/products.csv
export IMPLICIT_PREFERENCES_DATASET_PATH=data/implicit_preferences.csv
export NUM_ITEMS_PER_USER=3
export ALPHA=40.0

# Submit recommendation modeling job to the Spark standalone cluster
spark-submit \
  --master spark://127.0.0.1:7077 \
  --deploy-mode client \
  --num-executors 2 \
  --total-executor-cores 2 \
  --executor-memory 2g \
  --conf spark.sql.shuffle.partitions=4 \
  src/implicit_pipeline.py
</pre></code>

#### Executing the Explicit ML Pipeline on a Standalone Cluster
To execute the explicit feedback tracking module, declare your operational constraints and submit your workflow runtime to your standalone master node:

<pre><code>#!/bin/bash

source .venv/bin/activate

# Configure pipeline environment parameters
export MODEL_PATH=models/explicit_als_model
export USERS_DATASET_PATH=data/users.csv
export PRODUCTS_DATASET_PATH=data/products.csv
export EXPLICIT_PREFERENCES_DATASET_PATH=data/explicit_preferences.csv
export NUM_ITEMS_PER_USER=3

# Submit factorization modeling job to the Spark standalone cluster
spark-submit \
  --master spark://127.0.0.1:7077 \
  --deploy-mode client \
  --num-executors 2 \
  --total-executor-cores 2 \
  --executor-memory 2g \
  --conf spark.sql.shuffle.partitions=4 \
  src/explicit_pipeline.py
</pre></code>

## Automated Testing

Automated checking procedures run on top of the `pytest` engine. By configuring a custom `pythonpath` within your project manifest, namespace crashes are avoided entirely.

### Test Environment Lifecycle (`conftest.py`)
To prevent the overhead of spinning up a heavy Java Virtual Machine (JVM) instance for every individual test function, the framework includes a global `tests/conftest.py` configuration. This script establishes a shared, test-scoped `SparkSession` fixture optimized for PySpark 4.2.0 local, isolated text execution (e.g., using `local` threads and memory channel execution setups). 

The pipeline modules inject this pre-configured testing session into the primary engine runner (`run_pipeline`) alongside isolated `tmp_path` fixtures that construct mock CSV data on-the-fly. This guarantees that automated test suites execute safely in-memory without affecting your local workspace configurations or altering active production cluster tasks.

To execute the entire explicit and implicit engineering evaluation suite:

<pre><code>poetry run pytest -v</pre></code>
