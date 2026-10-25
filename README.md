# Customer Preference ML Pipeline

A scalable, production-grade PySpark Machine Learning project that leverages alternating least squares (ALS) matrix factorization to deliver customer product recommendations. The engine implements two distinct algorithmic training paradigms—**Explicit Feedback Factorization** and **Implicit Interaction Confidence Mapping**—using a highly decoupled, maintainable, and fully testable architecture.

---

## 📁 Project Structure

The project follows a standard Python project layout with a clean separation between source code (`src/`), automated tests (`tests/`), and environment configuration management (`pyproject.toml` managed via Poetry).

```text
.
├── README.md
├── data/
│   ├── explicit_preferences.csv     # Structured explicit review datasets (1-5 star ratings)
│   ├── implicit_preferences.csv     # Raw behavior event streams (page views, click counts)
│   ├── products.csv                 # Product item catalogue metadata registry
│   └── users.csv                    # Customer profiles and membership tiers
├── models/                          # Local directory target for serialized ML model weights
├── pyproject.toml                   # Project dependencies and tool suite configurations
├── src/
│   └── customer_preference_ml/
│       ├── __init__.py
│       ├── explicit_pipeline.py     # Explicit rating matrix execution script
│       └── implicit_pipeline.py     # Implicit engagement confidence tracking script
└── tests/
    └── customer_preference_ml/      # Mirror testing tree layout (free of structural namespace bugs)
        ├── test_explicit_pipeline.py
        └── test_implicit_pipeline.py
```

---

## 🧠 Architectural Overview & Theory

The framework isolates recommendations based on the core quality of data captured from customers:

| Feature / Metric | Explicit Pipeline (`explicit_pipeline.py`) | Implicit Pipeline (`implicit_pipeline.py`) |
| :--- | :--- | :--- |
| **Data Characteristic** | Direct, quantifiable user declarations (e.g., star ratings, thumbs up). | Passive consumer signals (e.g., clicks, total views, purchase volumes). |
| **Target Variable Column** | `rating` (Bound, linear scoring ranges). | `interaction_count` (Unbound non-negative values representing intensity). |
| **Algorithmic Mechanics** | Matrix completion to minimize structural square error errors on **known entries**. | Factorizes the *entire baseline transaction matrix* converting actions to binary preferences with a dynamic confidence adjustment ($\alpha$). |
| **Model Evaluation** | Validated via `RegressionEvaluator` metrics (**Root Mean Squared Error - RMSE**) against an un-trained test partition split. | Validated over the continuous global preference spectrum; relies heavily on hyper-parameter stability constraints. |

### 1. Explicit Pipeline Architecture
Configures the Spark ALS estimator with `implicitPrefs=False`. It divides user actions using an 80/20 random split window to build out explicit predictions. Rows where user metadata matches product parameters are dynamically compiled into an actionable relational report including properties like `membership_tier`.

### 2. Implicit Pipeline Architecture
Configures the Spark ALS estimator with `implicitPrefs=True`. Rather than fitting specific rating integers, it treats tracking numbers as interaction strength values. The baseline calculation is expanded by an **alpha ($\alpha$) modifier parameter**, mapping click frequencies to mathematical confidence variables prior to model weight persistence.

---

## ⚙️ Engineering & Code Restructuring Highlights

Both modules have been explicitly restructured to align with enterprise production-grade design principles:
* **Zero Global Side Effects on Import:** The script compilation phases have been stripped of direct `os.environ` dependency fetches. Environment variables are grouped inside isolated validation blocks inside `if __name__ == "__main__":` entry points.
* **Functional Decoupling:** Long procedural scripts have been cleanly split into modular functions with strict type hinting definitions (`get_spark_session`, `transform_recommendations`, and `run_pipeline`).
* **Test Isolation Ready:** Rather than spinning up a hardcoded local runtime, the primary engine runner (`run_pipeline`) accepts a pre-configured `SparkSession` context and configuration parameter `dict`. This allows local testing engines to inject lightweight memory allocations and temporary test files instantly.

---

## 🚀 Getting Started

### 1. Environment Prerequisite Setup
Ensure you have Python 3.12+ and Spark prerequisites configured locally. Initialize dependencies using Poetry:

```bash
# Install framework requirements, development dependencies, and pytest utilities
poetry install
```

### 2. Execution Run Paths via Cluster Deployments

Both pipelines run inside your target Spark Cluster environment. Ensure your local master worker (`spark://127.0.0.1:7077`) is running before job submission.

#### Executing the Implicit ML Pipeline
To run the implicit configuration track, activate your `.venv` container, pass your pipeline tuning configurations as shell environment variables, and route your job run directly via `spark-submit`:

```bash
#!/bin/bash
clear
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
  src/customer_preference_ml/implicit_pipeline.py
```

#### Executing the Explicit ML Pipeline
To execute the explicit feedback tracking module, declare your operational constraints and submit your workflow runtime to your standalone master node:

```bash
#!/bin/bash
clear
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
  src/customer_preference_ml/explicit_pipeline.py
```

---

## 🧪 Automated Testing

Automated checking procedures run on top of the `pytest` engine. By configuring a custom `pythonpath` within the project manifest, namespace crashes are avoided entirely. 

The test files make use of isolated, in-memory `SparkSession` configurations alongside `tmp_path` fixtures to construct mock CSV files on-the-fly, ensuring that automated test runs never alter local workspace datasets or conflict with your active production cluster tasks.

To execute the entire explicit and implicit engineering evaluation suite:

```bash
poetry run pytest -v
```
