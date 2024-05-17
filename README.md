# Pipeline Library

The Pipeline Library is a powerful and flexible tool designed to simplify the creation and management of machine learning pipelines. It provides a high-level interface for defining and executing pipelines, allowing users to focus on the core aspects of their machine learning projects. The library currently supports XGBoost models, with plans to expand support for more models in the future.

> [!WARNING]
> This library is in the early stages of development and is not yet ready for production use. The API and functionality may change without notice. Use at your own risk.

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
        "parameters": {
            "save_data_path": "ames_housing.pkl",
            "target": "SalePrice",
            "tracking": {
                "experiment": "ames_housing",
                "run": "baseline"
            }
        },
        "steps": [
            {
                "step_type": "GenerateStep",
                "parameters": {
                    "train_path": "examples/ames_housing/data/train.csv",
                    "predict_path": "examples/ames_housing/data/test.csv",
                    "drop_columns": [
                        "Id"
                    ]
                }
            },
            {
                "step_type": "TabularSplitStep",
                "parameters": {
                    "train_percentage": 0.7,
                    "validation_percentage": 0.2,
                    "test_percentage": 0.1
                }
            },
            {
                "step_type": "CleanStep"
            },
            {
                "step_type": "EncodeStep"
            },
            {
                "step_type": "ModelStep",
                "parameters": {
                    "model_class": "XGBoost"
                }
            },
            {
                "step_type": "CalculateMetricsStep"
            },
            {
                "step_type": "ExplainerDashboardStep",
                "parameters": {
                    "enable_step": false
                }
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

data = Pipeline.from_json("config.json").train()
```

3. Run the pipeline for inference using the following code:

```python
data = Pipeline.from_json("config.json").predict()
```

You can also set the prediction data as a DataFrame:

```python
data = Pipeline.from_json("config.json").predict(df)
```

This will use the DataFrame provided in code, not needing the `predict_path` file in the configuration parameters for the Generate step.

The library allows users to define custom steps for data generation, cleaning, and preprocessing, which can be seamlessly integrated into the pipeline.


## Performance and Memory Profiling

We've added pyinsytrument and memray as development dependencies for optimizing performance and memory usage of the library.
Refer to the tools documentation for usage notes:
- [memray](https://github.com/bloomberg/memray?tab=readme-ov-file#usage)
- [pyinstrument](https://pyinstrument.readthedocs.io/en/latest/guide.html#profile-a-python-cli-command)


## Contributing
Contributions to the Pipeline Library are welcome! If you encounter any issues, have suggestions for improvements, or want to add new features, please open an issue or submit a pull request on the GitHub repository.
