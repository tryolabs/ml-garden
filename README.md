# Pipeline Library

The purpose of this library is to create pipelines for ML as simple as possible. At the moment we support XGBoost models, but we are working to support more models.

This is an example of how to use the library to run an XGBoost pipeline:

We create a `train.json` file with the following content:

```json
{
    "custom_steps_path": "examples/ocf/",
    "save_path": "runs/xgboost_train.pkl",
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

The user can define custom steps to generate and clean their own data and use them in the pipeline. Then we can run the pipeline with the following code:

```python
import logging

from pipeline_lib.core import Pipeline

logging.basicConfig(level=logging.INFO)

Pipeline.from_json("train.json").run()
```