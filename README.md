# Pipeline Library

The Pipeline Library is a powerful and flexible tool designed to simplify the creation and management of machine learning pipelines. It provides a high-level interface for defining and executing pipelines, allowing users to focus on the core aspects of their machine learning projects. The library currently supports XGBoost models, with plans to expand support for more models in the future.

## Features

* Intuitive and easy-to-use API for defining pipeline steps and configurations
* Support for various data loading formats, including CSV and Parquet
* Flexible data preprocessing steps, such as data cleaning, feature calculation, and encoding
* Seamless integration with XGBoost for model training and prediction
* Hyperparameter optimization using Optuna for fine-tuning models
* Evaluation metrics calculation and reporting
* Explainable AI (XAI) dashboard for model interpretability
* Extensible architecture for adding custom pipeline steps

## Installation

To install the Pipeline Library, you need to have Python 3.9 or higher and Poetry installed. Follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/tryolabs/pipeline-lib.git
   ```

2. Navigate to the project directory:

    ```bash
    cd pipeline-lib
    ```

3. Install the dependencies using Poetry:

    ```bash
    poetry install
    ```

    If you want to include optional dependencies, you can specify the extras:

    ```bash
    poetry install --extras "xgboost"
    ```

    or

    ```bash
    poetry install --extras "all_models"
    ```

## Usage

Here's an example of how to use the library to run an XGBoost pipeline:

1. Create a `config.json` file with the following content:


```json
{
    "pipeline": {
        "name": "XGBoostTrainingPipeline",
        "description": "Training pipeline for XGBoost models.",
        "steps": [
            {
                "step_type": "GenerateStep",
                "parameters": {
                    "train_path": "examples/ocf/data/trainset_forecast.parquet",
                    "predict_path": "examples/ocf/data/testset_forecast.parquet"
                }
            },
            {
                "step_type": "CleanStep",
                "parameters": {
                    "drop_na_columns": [
                        "average_power_kw",
                        "diffuse_radiation"
                    ],
                    "drop_ids": {
                        "ss_id": [
                            7759,
                            7061
                        ]
                    }
                }
            },
            {
                "step_type": "CalculateFeaturesStep",
                "parameters": {
                    "datetime_columns": [
                        "date"
                    ],
                    "features": [
                        "year",
                        "month",
                        "day",
                        "hour",
                        "minute"
                    ]
                }
            },
            {
                "step_type": "TabularSplitStep",
                "parameters": {
                    "train_percentage": 0.95
                }
            },
            {
                "step_type": "FitModelStep",
                "parameters": {
                    "model_class": "XGBoostModel",
                    "target": "average_power_kw",
                    "drop_columns": [
                        "ss_id",
                        "operational_at",
                        "total_energy_kwh"
                    ],
                    "model_params": {
                        "max_depth": 12,
                        "eta": 0.12410097733370863,
                        "objective": "reg:squarederror",
                        "eval_metric": "mae",
                        "n_jobs": -1,
                        "n_estimators": 40,
                        "min_child_weight": 7,
                        "subsample": 0.8057743223537057,
                        "colsample_bytree": 0.6316852278944352,
                        "early_stopping_rounds": 10
                    },
                    "save_path": "model_forecast.joblib"
                }
            },
            {
                "step_type": "PredictStep",
                "parameters": {}
            },
            {
                "step_type": "CalculateTrainMetricsStep",
                "parameters": {}
            }
        ]
    }
}
```

2. Run the pipeline in train mode using the following code:

```python
import logging

from pipeline_lib import Pipeline

logging.basicConfig(level=logging.INFO)

data = Pipeline.from_json("config.json").run(is_train=True)
```

3. To run it in prediction mode you have to set `is_train` to `False`.

```python
data = Pipeline.from_json("config.json").run(is_train=False)
```

The library allows users to define custom steps for data generation, cleaning, and preprocessing, which can be seamlessly integrated into the pipeline.

## Contributing
Contributions to the Pipeline Library are welcome! If you encounter any issues, have suggestions for improvements, or want to add new features, please open an issue or submit a pull request on the GitHub repository.
