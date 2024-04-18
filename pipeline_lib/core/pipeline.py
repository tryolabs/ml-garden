from __future__ import annotations

import glob
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Optional

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow.data import from_pandas
from mlflow.tracking import MlflowClient

from pipeline_lib.core.data_container import DataContainer
from pipeline_lib.core.model_registry import ModelRegistry
from pipeline_lib.core.random_state_generator import initialize_random_state
from pipeline_lib.core.step_registry import StepRegistry
from pipeline_lib.core.steps import PipelineStep


class Pipeline:
    """Base class for pipelines."""

    _step_registry = {}
    logger = logging.getLogger("Pipeline")
    step_registry = StepRegistry()
    model_registry = ModelRegistry()

    KEYS_TO_SAVE = [
        "model",
        "encoder",
        "_generate_step_dtypes",
        "explainer",
    ]

    def __init__(
        self,
        save_data_path: str,
        target: str,
        columns_to_ignore_for_training: Optional[list[str]] = None,
    ):
        self.data = DataContainer()
        self.steps = []
        self.save_data_path = save_data_path
        self.data.target = target
        self.data.prediction_column = f"{target}_prediction"
        self.data.columns_to_ignore_for_training = columns_to_ignore_for_training or []
        self.config = {}

    def add_steps(self, steps: list[PipelineStep]):
        """Add steps to the pipeline."""
        self.steps.extend(steps)

    def run(self, is_train: bool, save: bool = True) -> DataContainer:
        """Run the pipeline on the given data."""

        if is_train:
            steps_to_run = [step for step in self.steps if step.used_for_training]
            self.logger.info("Training the pipeline")
        else:
            self.data.update(DataContainer.from_pickle(self.save_data_path))
            steps_to_run = [step for step in self.steps if step.used_for_prediction]
            self.logger.info("Predicting with the pipeline")

        self.data.is_train = is_train
        for i, step in enumerate(steps_to_run):
            start_time = time.time()
            log_str = f"Running {step.__class__.__name__} - {i + 1} / {len(steps_to_run)}"
            Pipeline.logger.info(log_str)

            data = step.execute(self.data)

            Pipeline.logger.info(f"{log_str} done. Took: {time.time() - start_time:.2f}s")

        if is_train:
            data.save(self.save_data_path, keys=self.KEYS_TO_SAVE)

        if save:
            self.save_run(data)

        return data

    def train(self) -> DataContainer:
        """Run the pipeline on the given data."""
        return self.run(is_train=True)

    def predict(self) -> DataContainer:
        """Run the pipeline on the given data."""
        return self.run(is_train=False)

    @classmethod
    def _validate_configuration(cls, config: dict[str, Any]) -> None:
        if "parameters" not in config["pipeline"]:
            raise ValueError("Missing pipeline parameters section in the config file.")

        if "save_data_path" not in config["pipeline"]["parameters"]:
            raise ValueError(
                "A path for saving the data must be provided. Use the `save_data_path` attribute "
                'of the pipeline parameters" section in the config.'
            )

        if "target" not in config["pipeline"]["parameters"]:
            raise ValueError(
                "A target column must be provided. Use the `target` attribute of the pipeline"
                ' "parameters" section in the config.'
            )

    @classmethod
    def from_json(cls, path: str) -> Pipeline:
        """Load a pipeline from a JSON file."""
        # check file is a json file
        if not path.endswith(".json"):
            raise ValueError(f"File {path} is not a JSON file")

        with open(path, "r") as config_file:
            config = json.load(config_file)

        Pipeline._validate_configuration(config)

        custom_steps_path = config.get("custom_steps_path")
        if custom_steps_path:
            cls.step_registry.load_and_register_custom_steps(custom_steps_path)

        pipeline = Pipeline(
            save_data_path=config["pipeline"]["parameters"]["save_data_path"],
            target=config["pipeline"]["parameters"]["target"],
            columns_to_ignore_for_training=config["pipeline"]["parameters"].get(
                "columns_to_ignore_for_training", []
            ),
        )
        pipeline.config = config

        if "seed" in config["pipeline"]["parameters"]:
            seed = config["pipeline"]["parameters"]["seed"]
        else:
            seed = 42
        initialize_random_state(seed)

        steps = []

        for step_config in config["pipeline"]["steps"]:
            step_type = step_config["step_type"]
            parameters = step_config.get("parameters", {})

            Pipeline.logger.info(
                f"Creating step {step_type} with parameters: \n {json.dumps(parameters, indent=4)}"
            )

            # change model from string to class
            if step_type == "FitModelStep":
                model_class_name = parameters.pop("model_class")
                model_class = cls.model_registry.get_model_class(model_class_name)
                parameters["model_class"] = model_class

            step_class = cls.step_registry.get_step_class(step_type)
            step = step_class(**parameters)
            steps.append(step)

        pipeline.add_steps(steps)
        return pipeline

    def log_experiment(
        self,
        data: DataContainer,
        experiment_name: str,
        run_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """
        Log the pipeline run to MLflow.

        Parameters
        ----------
        data : DataContainer
            The data container object containing the pipeline data.
        experiment_name : str
            The name of the MLflow experiment.
        run_name : str, optional
            The name of the MLflow run. If not provided, a default run name will be generated
            based on the pipeline class name, mode (train or predict), and current timestamp.
        dataset_name : str, optional
            The name of the dataset to be logged as an input to MLflow. If provided, the input
            data will be logged with the specified dataset name.
        description : str, optional
            The description of the MLflow run.

        Returns
        -------
        None

        Notes
        -----
        This function logs various aspects of the pipeline run to MLflow, including:
        - Top-level pipeline parameters from the configuration
        - Step-level parameters from the configuration
        - Input data (if `dataset_name` is provided)
        - Training metrics (if available in the `data` object)
        - Trained model (if available in the `data` object)

        The function sets the MLflow experiment using the provided `experiment_name` and starts
        a new MLflow run with the specified `run_name` (or a default run name if not provided).

        The top-level pipeline parameters and step-level parameters are logged as MLflow parameters.
        If `dataset_name` is provided, the input data is logged as an MLflow input using the
            specified name.
        If training metrics are available in the `data` object, they are logged as MLflow metrics.
        If a trained model is available in the `data` object, it is logged as an MLflow artifact.

        Examples
        --------
        >>> data = DataContainer(...)
        >>> pipeline.log_experiment(data, experiment_name='my_experiment', run_name='run_1',
            dataset_name='input_data')
        """

        def log_params_from_config(config):
            # Log top-level parameters
            for key in ["name", "description", "save_data_path"]:
                if key in config["pipeline"]:
                    mlflow.log_param(f"pipeline.{key}", config["pipeline"][key])

            # Log step-level parameters
            for i, step in enumerate(config["pipeline"]["steps"]):
                mlflow.log_param(f"pipeline.steps_{i}.step_type", step["step_type"])
                for key, value in step.get("parameters", {}).items():
                    mlflow.log_param(f"pipeline.steps_{i}.parameters.{key}", value)

        def plot_feature_importance(df: pd.DataFrame) -> None:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(df["feature"], df["importance"])
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")
            ax.set_title("Feature Importance")
            # add grid
            ax.grid(axis="x")
            plt.tight_layout()
            mlflow.log_figure(fig, "feature_importance.png")

        mlflow.set_experiment(experiment_name)

        if not run_name:
            mode_name = "train" if data.is_train else "predict"
            run_name = (
                f"{self.__class__.__name__}_{mode_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            self.logger.info(f"Run name not provided. Using default run name: {run_name}")

        with mlflow.start_run(run_name=run_name):

            mlflow.set_tag("name", self.config["pipeline"]["name"])
            mode_str = "train" if data.is_train else "prediction"
            mlflow.set_tag("mode", mode_str)

            log_params_from_config(self.config)

            if description:
                mlflow.set_tag("mlflow.note.content", description)

            if dataset_name:
                self.logger.info(f"Logging input data to MLflow with dataset name: {dataset_name}")
                mlflow.log_input(from_pandas(data.raw), dataset_name)

            # Log prediction metrics
            if data.metrics:
                self.logger.debug("Logging prediction metrics to MLflow")
                if data.is_train:
                    for dataset_name, metrics in data.metrics.items():
                        for metric_name, metric_value in metrics.items():
                            mlflow.log_metric(f"{dataset_name}_{metric_name}", metric_value)
                else:
                    mlflow.log_metrics(data.metrics["prediction"])

            # Log the model
            if data.is_train and data.model:
                self.logger.debug("Logging the model to MLflow")
                mlflow.sklearn.log_model(data.model, artifact_path="model")

            # Save feature importance as a png artifact
            if data.feature_importance is not None:
                self.logger.debug("Plotting feature importance for MLflow")
                plot_feature_importance(data.feature_importance)

            if self.config:
                self.logger.debug("Logging pipeline configuration to MLflow as a JSON file")
                # convert model_class to string
                fit_step = next(
                    step
                    for step in self.config["pipeline"]["steps"]
                    if step["step_type"] == "FitModelStep"
                )
                fit_step["parameters"]["model_class"] = fit_step["parameters"][
                    "model_class"
                ].__name__
                mlflow.log_dict(self.config, "config.json")

            # save data container pickle as an artifact
            if self.save_data_path:
                self.logger.debug("Logging the data container to MLflow")
                # change name to data_container.pkl
                mlflow.log_artifact(self.save_data_path, artifact_path="data")

    @classmethod
    def from_mlflow_experiment(cls, run_id: str) -> Pipeline:
        """Load a pipeline from an MLflow run."""
        # Create an instance of MlflowClient

        client = MlflowClient()
        run = client.get_run(run_id)
        artifacts_dir = run.info.artifact_uri

        print(artifacts_dir)

        # Download the config.json artifact
        config_path = os.path.join(artifacts_dir, "config.json")
        config_path = config_path.replace("file://", "")  # Remove the "file://" prefix if present

        # Search for the data container file in the "data" folder
        data_folder_path = os.path.join(artifacts_dir, "data")
        data_folder_path = data_folder_path.replace("file://", "")
        data_container_files = glob.glob(os.path.join(data_folder_path, "*.pkl"))

        if len(data_container_files) == 0:
            raise FileNotFoundError("No data container file found in the 'data' folder.")
        elif len(data_container_files) > 1:
            raise ValueError("Multiple data container files found in the 'data' folder.")

        data_container_path = data_container_files[0]

        # Use the existing from_json function to load the pipeline
        pipeline = cls.from_json(config_path)
        pipeline.save_data_path = data_container_path
        return pipeline

    def save_run(
        self,
        data: DataContainer,
        parent_folder: str = "runs",
    ) -> None:
        """Save the pipeline run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{self.__class__.__name__}_{timestamp}"
        run_folder = os.path.join(parent_folder, folder_name)

        # Create the run folder
        os.makedirs(run_folder, exist_ok=True)

        # Save the JSON configuration
        with open(os.path.join(run_folder, "pipeline_config.json"), "w") as f:
            json.dump(self.config, f, indent=4, cls=CustomJSONEncoder)

        # Save the training metrics
        if data.metrics:
            with open(os.path.join(run_folder, "metrics.json"), "w") as f:
                json.dumps({k: str(v) for k, v in data.metrics.items()}, indent=4)

        self.logger.info(f"Pipeline run saved to {run_folder}")

    def __str__(self) -> str:
        step_names = [f"{i + 1}. {step.__class__.__name__}" for i, step in enumerate(self.steps)]
        return f"{self.__class__.__name__} with steps:\n" + "\n".join(step_names)

    def __repr__(self) -> str:
        """Return an unambiguous string representation of the pipeline."""
        step_names = [f"{step.__class__.__name__}()" for step in self.steps]
        return f"{self.__class__.__name__}({', '.join(step_names)})"


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, type):
            return obj.__name__
        return super().default(obj)
