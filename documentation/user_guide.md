# MLGarden User Guide

# Introduction

MLGarden is a versatile and intuitive pipeline library designed to simplify the creation and management of machine learning projects. With a high-level interface for defining and executing pipelines, MLGarden empowers users to focus on the core aspects of their machine learning projects without getting bogged down in the details.

Currently supporting XGBoost models for regression tasks based on tabular data, MLGarden offers a range of features to streamline your machine learning workflow. From intuitive API for defining pipeline steps to support for various data loading formats and flexible data preprocessing steps, MLGarden has everything you need to build and train powerful machine learning models with ease.

In addition to seamless integration with XGBoost for model training and prediction, MLGarden also provides hyperparameter optimization using Optuna, evaluation metrics calculation, and an Explainable AI (XAI) dashboard for model interpretability. The extensible architecture of the library allows users to add custom pipeline steps and tailor the pipeline to their specific needs.

This user guide is designed to help you harness the full potential of MLGarden's capabilities. Whether you are a beginner looking to kickstart your machine learning journey or an experienced practitioner seeking to streamline your workflow, this guide will walk you through the key features and functionalities of the pipeline library.

## Key components of the pipeline:

_Pipeline Steps:_ These are the predefined steps that are connected to each other to pass information through the pipeline in a data container. Users can define and configure these steps to create a customized pipeline for their machine learning projects.

_Config File:_ Users generate a config file to set the pipeline steps to use and specify their parameters. This config file serves as a roadmap for the pipeline, outlining the sequence of steps and their configurations.

_Data Container:_ The Data Container serves as a central repository for storing and passing all the essential information, data, and intermediate results through the pipeline steps. It ensures smooth communication and transfer of data between the various components of the pipeline, enabling effective data processing, transformation, and analysis at each stage of the machine learning workflow.

# Getting started

## **Installation**

To install the Pipeline Library, you need to have `Python 3.9` or higher and `Poetry` installed. Follow these steps:

1. Clone the repository:

```python
git clone https://github.com/tryolabs/ml-garden.git
```

2. Navigate to the project directory:

```python
cd ml-garden
```

3. Install the dependencies using Poetry:

```python
poetry install
```

If you want to include optional dependencies, you can specify the extras:

```python
poetry install --extras "xgboost"
```

or

```python
poetry install --extras "all_models"
```

## Quick start

Here you can see an example of a `config.json` file to run an XGBoost pipeline (for more detailed explanation please refer to [Generating and Executing Pipelines](#pipelines):

```json
{
"pipeline": {
	"name": "MyTrainingPipeline",
	"description": "My amazing project",
	"parameters": {
            "save_data_path": "pipeline.pkl",
            "target": "target_column"
		}
	},
	"steps":[
	{
        "step_type": "GenerateStep",
            "parameters": {
                "train_path": "path/to/train/data.csv",
                "test_path": "path/to/test/data.csv","predict_path": "path/to/predict/data.csv",
                "drop_columns": [
	            "column_to_drop"
                    ]
                }

		},
		{
		"step_type": "TabularStep",
		"parameters":{
			"train_percentage": 0.7,
			"validation_percentage":0.2,
			"test_percentage":0.1
		},
		{
		"step_type": "CleanStep",
			}
		},
    {
    "step_type": "EncodeStep",
    "parameters": {}
     },
     {
     "step_type": "ModelStep",
        "parameters": {
           "model_class": "XGBoostModel",
       }
     },
     {
     "step_type": "CalculateMetricsStep",
     },
     {
     "step_type": "ExplainerDashboardStep",
         "parameters": {
         "enable_step": true
         }
     }
	}
	]
}
```

You can run the pipeline in train mode using the following code

```python
import logging

from ml-garden import Pipeline

logging.basicConfig(level=logging.INFO)

data = Pipeline.from_json("config.json").train()
```

To run the pipeline for inference you can use the following code

```python
data = Pipeline.from_json("config.json").predict()
```

You can also set the prediction data as a DataFrame:

```python
data = Pipeline.from_json("config.json").predict(df)
```

This will use the DataFrame provided in code, not needing the `predict_path` file in the configuration parameters for the Generate step.

# Steps

A list of all steps you can define in the `config.json` can be found in the following table

| step                   | training | prediction | description                                                                                               | parameters                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ---------------------- | -------- | ---------- | --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| CalculateFeaturesStep  | ✅       | ✅         | Extract date features from columns that contain dates                                                     | `datetime_columns: Optional[List[str]]` List of columns, that contain dates, `features: Optional[List[str]]` List of features, that should be extracted from `datetime_columns`. Choose from "year”, "month", "day", "hour", "minute", "second", "weekday", "dayofyear”                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| CalculateMetricsStep   | ✅       | ❌         | Calculate a set of regression metrics: MAE, RMSE, R², mean error, Median Absolute Error                   | `mape_threshold: float = 0.01` Threshold for MAPE calculation. Default value is set to 0.1.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| CalculateReportsStep   | ✅       | ❌         | Calculate SHAP values and feature importances                                                             | `max_samples: int = 1000` Maximum number of samples used for the calculations of the report. Default value is set to 1000.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |     |
| CleanStep              | ✅       | ✅         | Clean the data: fill missing values, remove outliers, convert types, drop missing values, drop rows by id | `fill_missing: Optional[dict]` Dictionary, where the key defines the column and the value the value that should be filled in for a missing value, `remove_outliers: Optional[dict]` Dictionary, where the key defines the column and the value, the method that should be used to remove outliers. Choose between: clip: clips outliers by 0.25/0.75 percentile, drop: drops outliers that are lower/higher than 0.25/0.75 percentile, `convert_dtypes: Optional[dict]` Dictionary, where the key defines the column and the value of the type that column should be converted to, `drop_na_columns: Optional[list]` List of columns, that define rows to be dropped by missing values of these columns, `drop_ids: Optional[dict]` Dictionary, where the key defines the column and the value the value of that column based on which rows are dropped, `filter: Optional[dict]` Dictionary, where the key defines the column and the value a rule based on which the data is filtered. The rule must be an expression interpretable by a Pandas query. |
| EncodeStep             | ✅       | ✅         | Encode categorical features using ordinal encoding. Numerical features are not affected by this step.     | `cardinality_threshold: int = 5` Threshold to split features in low and high cardinality. Default set to 5, `feature_encoders: Optional[dict]` Dictiionary, where the key contains the feature and the value the encoder that should be used. Choose between “OrdinalEncoder” and “TargetEncoder”. If feature_encoders is not provided default encoders based on cardinality are used. features with low cardinality use the `OrdinalEncoder`, features with high cardinality use the `TargetEndoder`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ExplainerDashboardStep | ✅       | ❌         | Calculate SHAP values.                                                                                    | `max_samples: int = 1000` Maximum number of samples used for the calculations. Default set to 1000, `X_background_samples: int = 100` length of background dataset enable_step: bool = True If set to False this step is not executed.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ModelStep              | ✅       | ✅         | Fit the model.                                                                                            | `model_class: Type[Model]` The model to be used for fitting, `model_parameters: Optional[dict]` Dictionary containing the model parameters, `optuna_params: Optional[dict]` Dictionary containing parameters for Optuna, `save_path: Optional[str]` Path where results are saved.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| GenerateStep           | ✅       | ✅         | Generate the data container. Supported input data formats are csv and parquet                             | `train_path: Optional[str]` Path to training datatest_path: Optional[str] Path to test data, `predict_path: Optional[str]` Path to data used for prediction, `drop_columns: Optional[list[str]]` Columns to be dropped from the dataframeoptimize_dtypes: bool If True categorical columns are converted to type “category”,`optimize_dtypes_skip_cols: Optional[list[str]]` Columns to be skipped in category detection and conversion.\*\*kwargs Additional parameters to read the data.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| TabularSplitStep       | ✅       | ❌         | Split the data for training, validation, and test sets.                                                   | `train_percentage: float` Percentage of the data used for training (between 0 and 1), `validation_percentage: Optional[float]` Percentage of the data used for validation (between 0 and 1), `test_percentage: Optional[float]` Percentage of the data used for testing (between 0 and 1). If not provided no testset is generated, `group_by_columns: Optional[list[str]]` Columns defining the groups by which the splits will be performed.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |

Table 1: List of all available steps, their description, and parameters.

# Using MLGarden

## Configuration File

The configuration file defines the steps that should be executed by the pipeline. The configuration is given as a JSON file. This file has two keys: `pipeline` and `steps`. The `pipeline` key contains general information about the pipeline, like the path to save the results, the target column, and columns that should be ignored for training. The `step` key contains a list of steps to be executed. Each step is described by the `step_type` and the `parameters`. A list of all available steps and parameters can be found in the previous section

```json
{
"pipeline": {
    "name": "MyTrainingPipeline",
    "description": "My amazing project",
        "parameters": {
            "save_data_path": "pipeline.pkl",
            "target": "target_column"
		}
	},
	"steps":[
	{
            "step_type": "GenerateStep",
            "parameters": {
                "train_path": "path/to/train/data.csv",
                "test_path": "path/to/test/data.csv",
                "predict_path": "path/to/predict/data.csv",
                "drop_columns": [
	            "column_to_drop"
                    ]
                }

		},
		{
		"step_type": "TabularStep",
		"parameters":{
			"train_percentage": 0.8
		},
		{
		"step_type": "CleanStep",
		"parameters":{
			"filter": "column_name > 1000"
			"drop_na_columns": [
				"column_name1",
				"column_name2"
				]
			"drop_ids": {
				"id": [1, 100, 505]
			}
		},
		{
    "step_type": "CalculateFeaturesStep",
    "parameters": {
       "datetime_columns": ["date"],
       "features": [
          "month",
          "day",
          "hour"
          ]
      }
    },
    {
    "step_type": "EncodeStep",
    "parameters": {}
     },
     {
     "step_type": "ModelStep",
        "parameters": {
           "model_class": "XGBoostModel",
           "model_parameters": {
             "max_depth": 5,
             "eta": 0.1,
             "objective": "reg:squarederror",
             "eval_metric": "made",
             "n_jobs": -1,
             "n_estimators": 50,
             "min_child_weight": 1,
             "subsample": 1,
             "colsample_bytree": 1,
             "early_stopping_rounds": 20,
         }
       }
     },
     {
     "step_type": "PredictStep",
        "parameters": {}
     },
     {
     "step_type": "CalculateMetricsStep",
         "parameters": {}
     },
     {
     "step_type": "ExplainerDashboardStep",
         "parameters": {}
     }
	}
	]
}
```

More examples of configuration files, including hyperparameter tuning, can be found in the [examples](https://github.com/tryolabs/pipeline-lib/tree/main/examples/ames_housing/configs) folder.

## Generating and Executing Pipelines{#pipelines}

From this configuration file, we can directly start a pipeline that executes the defined steps with the desired parameters.

```python
import logging

from ml-garden import Pipeline

logging.basicConfig([level=logging.INFO](http://level=logging.info/))

data = Pipeline.from_json("config.json").train()
```

The Pipeline can be called in `train` or `predict` mode. The steps performed in each mode are illustrated in Table 1. By providing the defined `config.json` you define the steps and parameters that should be used by the pipeline. The results are stored in the folder `runs`, which will be created during execution, if it doesn’t exist. Within this folder a new folder for each pipeline execution will be created, in which the model configurations are stored. If the step `CalculateMetricsStep` is included, the pipeline calculates a set of evaluation metrics (for details, please refer to Table 1), which are also stored in that same folder.

The pipeline execution returns a DataContainer object, in the above example, called `data`. This object contains the raw input data as a Pandas dataframe, which can be accessed using `data.raw`. Depending on the steps included in the pipeline the DataContainer contains different objects, that are defined and calculated during the execution of the steps. If all implemented steps are executed, the DataContainer contains the following objects.

```python
{
    "target": "str",
    "prediction_column": "str",
    "columns_to_ignore_for_training": "list",
    "is_train": "bool",
    "_generate_step_dtypes": "dict",
    "raw": "DataFrame",
    "flow": "DataFrame",
    "split_indices": "dict",
    "train": "DataFrame",
    "validation": "DataFrame",
    "test": "DataFrame",
    "X_train": "DataFrame",
    "y_train": "Series",
    "encoder": "ColumnTransformer",
    "X_validation": "DataFrame",
    "y_validation": "Series",
    "X_test": "DataFrame",
    "y_test": "Series",
    "model": "XGBoost",
    "metrics": "dict",
    "explainer": "RegressionExplainer"
}
```

**Hyperparameter Tuning**

Hyperparameter tuning is supported using [Optuna](https://optuna.org/). If hyperparameter tuning is used in the `FitModelStep`, the results can be visualized in the browser using the following command. Note, this assumes the setting of `"storage": "sqlite:///db.sqlite3"` in the Optuna parameters of the `FitModelStep`

```bash
optuna-dashboard sqlite:///db.sqlite3
```

This will allow you to see the dashboard on `http://localhost:8080/`. For more details, please refer to the [Optuna documentation](https://optuna-dashboard.readthedocs.io/en/latest/getting-started.html). An example configuration including hyperparamter tuning can be found in the [examples](https://github.com/tryolabs/pipeline-lib/tree/main/examples/ames_housing/configs) folder.

## **Explainer Dashboard**

The pipeline integrates the [explainerdashboard](https://explainerdashboard.readthedocs.io/en/latest/) library, which allows to visualize and explain the results. It includes visualization of feature importances and different tools of explainable AI, like SHAP values, permutation importances, interaction effects, partial dependence plots, and more. The explainer is stored in the pipeline, if this step is enabled. The pipeline is saved in a compressed format, it can be decompressed and the dashboard can be accessed as shown in the following code block.

```python
import pickle
import zipfile
from explainerdashboard import ExplainerDashboard

file_name = pipeline.pkl"

with zipfile.ZipFile(f"{file_name}.zip", "r") as zip_file:
    zip_file.extractall()

with open(file_name, 'rb') as f:
    pipeline = pickle.load(f)
    ExplainerDashboard(pipeline["explainer"]).run()
```

The Dashboard is then available under [`http://localhost:8050`](http://localhost:8050/).

The explainerdashboard can be disabled using `enable: false` in the configuration in the `ExplainerDashboardType`

```python
{
    "step_type":
    "ExplainerDashboardStep",
            "parameters": {
                 "enable_step": false,
    }
}
```

## **Experiment Tracking**

Experiment tracking is performed using [mlflow](https://mlflow.org/). The experiment name and the run name can be defined in the general parameters section of the `config.json` file.

```json
{
"pipeline": {
        "name": "MyTrainingPipeline",
        "description": "My amazing project",
        "parameters": {
            "save_data_path": "pipeline.pkl",
            "target": "target_column",
            "tracking": {
                "experiment": "experiment_name",
                "run": "run_name"
            }
        },
 "steps": [
 ...
		]
 }
```

The results can then seen in the browser using the command.

```bash
mlflow server --host 127.0.0.1 --port 8080
```

For more details, please refer to the [mlflow documentation](https://mlflow.org/docs/latest/index.html). Note, that if the parameters for `tracking` are not provided, the experiment is not tracked.

# Customizing the Library for Your Project

The pipeline can be customized to a specific project not only by selecting the steps to be performed. If you need to perform steps that are not available, you can add them to the folder `ml-garden/core/steps/custom_step.py` and then include them into your `config.json`. There you have to provide a “step_type”, which is simply the name of the class you define and the “parameters” of this class. The defined class needs to have an `execute` method, which executes the calculations to be done in the step. The custom step needs to inherit from the `PipelineStep`. It receives an object from the `DataContatiner` and returns it. The code snippet below shows the structure of a step.

```python
from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps.base import PipelineStep

class CustomStep(PipelineStep):
    used_for_prediction = True
    used_for_training = True

    def __init__(self)
	    # set the parameters needed

    def execute(self, data: DataContainer) -> DataContainer:
		  # perform calculations for this step

    return data
```

New models can be added to the `pipeline_lib/implementation/tabular` folder. Currently, the pipeline supports tabular data for regression problems. However, if you want to perform a different task, more sophisticated changes will be necessary.

If you think your work would be valuable for other projects, please consider contributing it to the library.
