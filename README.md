# ML-GARDEN

ml-garden is a pipeline library that simplifies the creation and management of machine learning projects. It offers a high-level interface for defining and executing pipelines, allowing users to focus on their projects without getting lost in details. It currently supports XGBoost models for regression tasks on tabular data, with plans to expand support for more models in the future.
The key components of the pipeline include Pipeline Steps, which are predefined steps connected to pass information through a data container; a Config File for setting pipeline steps and parameters; and a Data Container for storing and transferring essential data and results throughout the pipeline, facilitating effective data processing and analysis in machine learning projects.

> [!WARNING]
> Please be advised that this library is currently in the early stages of development and is not recommended for production use at this time. The API and functionality of the library may undergo changes without prior notice. Kindly use it at your own discretion and be aware of the associated risks.

## Features

- Intuitive and easy-to-use API for defining pipeline steps and configurations
- Support for various data loading formats, including CSV and Parquet
- Flexible data preprocessing steps, such as data cleaning, feature calculation, and encoding
- Seamless integration with XGBoost for model training and prediction
- Hyperparameter optimization using Optuna for fine-tuning models
- Evaluation metrics calculation and reporting
- Explainable AI (XAI) dashboard for model interpretability
- Extensible architecture for adding custom pipeline steps

## Installation

To install the Pipeline Library, you need to have Python 3.9 or higher and Poetry installed. Follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/tryolabs/ml-garden.git
   ```

2. Navigate to the project directory:

   ```bash
   cd ml-garden
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
          "drop_columns": ["Id"]
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

from ml_garden import Pipeline

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
