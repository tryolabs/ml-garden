# Pipeline Library

The Pipeline Library is designed to simplify the creation of machine learning pipelines. Currently, it supports XGBoost models, with plans to expand support for more models in the future.

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

1. Create a `train.json` file with the following content:


```json
{
    "custom_steps_path": "examples/ocf/",
    "pipeline": {
        "name": "XGBoostTrainingPipeline",
        "description": "Training pipeline for XGBoost models.",
        "steps": [
            {
                "step_type": "OCFGenerateStep",
                "parameters": {
                    "path": "examples/ocf/data/trainset_new.parquet"
                }
            },
            {
                "step_type": "OCFCleanStep",
                "parameters": {}
            },
            {
                "step_type": "TabularSplitStep",
                "parameters": {
                    "id_column": "ss_id",
                    "train_percentage": 0.95
                }
            },
            {
                "step_type": "XGBoostFitModelStep",
                "parameters": {
                    "target": "average_power_kw",
                    "drop_columns": [
                        "ss_id"
                    ],
                    "xgb_params": {
                        "max_depth": 12,
                        "eta": 0.12410097733370863,
                        "objective": "reg:squarederror",
                        "eval_metric": "mae",
                        "n_jobs": -1,
                        "n_estimators": 2,
                        "min_child_weight": 7,
                        "subsample": 0.8057743223537057,
                        "colsample_bytree": 0.6316852278944352
                    },
                    "save_model": true
                }
            }
        ]
    }
}
```

2. Run the pipeline using the following code:

```python
import logging

from pipeline_lib.core import Pipeline

logging.basicConfig(level=logging.INFO)

Pipeline.from_json("train.json").run()
```

The library allows users to define custom steps for generating and cleaning their own data, which can be used in the pipeline.
