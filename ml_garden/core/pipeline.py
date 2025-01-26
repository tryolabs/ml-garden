from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import mlflow

from ml_garden.core.constants import Task
from ml_garden.core.data_container import DataContainer
from ml_garden.core.model_registry import ModelRegistry
from ml_garden.core.random_state_generator import RandomStateManager
from ml_garden.core.step_registry import StepRegistry
from ml_garden.core.steps import PipelineStep
from ml_garden.core.steps.fit_model import ModelStep

if TYPE_CHECKING:
    import pandas as pd

    from ml_garden.core.steps import PipelineStep

# ruff: noqa: PLR0915 C901


class Pipeline:
    """Base class for pipelines."""

    _step_registry = {}
    logger = logging.getLogger("Pipeline")
    step_registry = StepRegistry()
    # TODO: model and step registries should be singletons, to avoid coupling Pipeline with all
    # the steps and models
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
        task: Task,
        columns_to_ignore_for_training: list[str] | None = None,
        tracking: dict | None = None,
    ) -> None:
        if not isinstance(task, Task):
            error_message = f"task must be an instance of Task enum, got {type(task)}"
            self.logger.error(error_message)
            raise TypeError(error_message)

        self.steps = []
        self.save_data_path = save_data_path
        self.target = target
        self.prediction_column = f"{target}_prediction"
        self.columns_to_ignore_for_training = columns_to_ignore_for_training or []
        self.tracking = tracking or {}
        self.config = {}
        self.task = task

    def _initialize_data(self) -> DataContainer:
        """Initialize the data container."""
        data = DataContainer()
        data.target = self.target
        data.prediction_column = self.prediction_column
        data.columns_to_ignore_for_training = self.columns_to_ignore_for_training
        data.task = self.task
        return data

    def add_steps(self, steps: list[PipelineStep]) -> None:
        """Add steps to the pipeline."""
        self.steps.extend(steps)

    def run(self, *, is_train: bool, df: pd.DataFrame | None = None) -> DataContainer:
        """Run the pipeline on the given data."""
        data = self._initialize_data()

        if is_train:
            steps_to_run = [step for step in self.steps if step.used_for_training]
            self.logger.info("Training the pipeline")
        else:
            loaded_data = DataContainer.from_pickle(self.save_data_path)
            if loaded_data is None:
                error_message = f"Failed to load data from the pickle file ({self.save_data_path})."
                self.logger.error(error_message)
                raise ValueError(error_message)
            data.update(loaded_data)
            if df is not None:
                data.raw = df
            steps_to_run = [step for step in self.steps if step.used_for_prediction]
            self.logger.info("Predicting with the pipeline")

        data.is_train = is_train

        for i, step in enumerate(steps_to_run):
            start_time = time.time()
            log_str = f"Running {step.__class__.__name__} - {i + 1} / {len(steps_to_run)}"
            Pipeline.logger.info(log_str)

            data = step.execute(data)

            Pipeline.logger.info(f"{log_str} done. Took: {time.time() - start_time:.2f}s")

        if is_train:
            data.save(self.save_data_path, keys=self.KEYS_TO_SAVE)

        if self.tracking and is_train:
            self.logger.info("Logging pipeline run to MLflow")
            self.log_experiment(data, **self.tracking)
            self.logger.info("Finished logging pipeline run to MLflow")

        return data

    def train(self) -> DataContainer:
        """Run the pipeline in training mode."""
        return self.run(is_train=True)

    def predict(self, df: pd.DataFrame | None = None) -> DataContainer:
        """Run the pipeline in inference mode.

        Parameters
        ----------
        df : Optional[pd.DataFrame], optional
            The input data to make predictions on. If not provided, the data is taken from the
            `predict_path` attribute of the GenerateStep in the configuration JSON.

        Returns
        -------
        DataContainer
            The data container object containing the pipeline data after running the pipeline in
            inference mode.
        """
        return self.run(is_train=False, df=df)

    @classmethod
    def _validate_configuration(cls, config: dict[str, Any]) -> None:
        if "parameters" not in config["pipeline"]:
            error_message = "Missing pipeline parameters section in the config file."
            cls.logger.error(error_message)
            raise ValueError(error_message)

        if "save_data_path" not in config["pipeline"]["parameters"]:
            error_message = (
                "A path for saving the data must be provided. Use the `save_data_path` attribute "
                'of the pipeline parameters" section in the config.'
            )
            cls.logger.error(error_message)
            raise ValueError(error_message)

        if "target" not in config["pipeline"]["parameters"]:
            error_message = (
                "A target column must be provided. Use the `target` attribute of the pipeline"
                ' "parameters" section in the config.'
            )
            cls.logger.error(error_message)
            raise ValueError(error_message)

    @classmethod
    def _validate_task(cls, task: str) -> Task:
        try:
            return Task(task.lower())
        except ValueError:
            supported_tasks = ", ".join(t.value for t in Task)
            error_message = f"Invalid task: {task}. Supported tasks are: {supported_tasks}"
            cls.logger.exception(error_message)
            raise ValueError(error_message) from None

    @classmethod
    def from_json(cls, path: str) -> Pipeline:
        """Load a pipeline from a JSON file."""
        # check file is a json file
        if not path.endswith(".json"):
            error_message = f"File {path} is not a JSON file"
            cls.logger.error(error_message)
            raise ValueError(error_message)

        with Path(path).open() as config_file:
            config = json.load(config_file)

        cls._validate_configuration(config)

        custom_steps_path = config.get("custom_steps_path")
        if custom_steps_path:
            cls.step_registry.load_and_register_custom_steps(custom_steps_path)

        task = cls._validate_task(config["pipeline"]["parameters"]["task"])

        pipeline = Pipeline(
            save_data_path=config["pipeline"]["parameters"]["save_data_path"],
            target=config["pipeline"]["parameters"]["target"],
            columns_to_ignore_for_training=config["pipeline"]["parameters"].get(
                "columns_to_ignore_for_training", []
            ),
            tracking=config["pipeline"]["parameters"].get("tracking"),
            task=task,
        )
        pipeline.config = config

        seed = config["pipeline"]["parameters"].get("seed", 42)
        RandomStateManager.initialize(seed)

        steps = []

        for step_config in config["pipeline"]["steps"]:
            step_type = step_config["step_type"]
            step_class = cls.step_registry.get_step_class(step_type)
            parameters = step_config.get("parameters", {})

            cls.logger.info(
                "Creating step %s with parameters: \n %s",
                step_type,
                json.dumps(parameters, indent=4),
            )

            # change model from string to class
            if issubclass(step_class, ModelStep):
                model_class_name = parameters.pop("model_class")
                model_class = cls.model_registry.get_model_class(model_class_name)
                parameters["model_class"] = model_class

                if pipeline.task not in model_class.TASKS:
                    error_message = (
                        f"The model class '{model_class_name}' is intended for '{model_class.TASK}'"
                        f" tasks, but the pipeline task is '{pipeline.task}'."
                    )
                    cls.logger.error(error_message)
                    raise ValueError(error_message)

            step = step_class(**parameters)
            steps.append(step)

        pipeline.add_steps(steps)
        return pipeline

    def log_experiment(
        self,
        data: DataContainer,
        experiment: str,
        run: str | None = None,
        description: str | None = None,
        tracking_uri: str | None = None,
    ) -> None:
        """
        Log the pipeline run to MLflow.

        Parameters
        ----------
        data : DataContainer
            The data container object containing the pipeline data.
        experiment : str
            The name of the MLflow experiment.
        run : str, optional
            The name of the MLflow run. If not provided, a default run name will be generated
            based on the pipeline class name, mode (train or predict), and current timestamp.
        description : str, optional
            The description of the MLflow run.
        tracking_uri : str, optional
            The URI of the MLflow tracking server. If not provided, the default tracking URI will
            be used.

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

        The function sets the MLflow experiment using the provided `experiment` and starts
        a new MLflow run with the specified `run` (or a default run name if not provided).

        The top-level pipeline parameters and step-level parameters are logged as MLflow parameters.
        If `dataset_name` is provided, the input data is logged as an MLflow input using the
            specified name.
        If training metrics are available in the `data` object, they are logged as MLflow metrics.
        If a trained model is available in the `data` object, it is logged as an MLflow artifact.

        Examples
        --------
        >>> data = DataContainer(...)
        >>> pipeline.log_experiment(data, experiment='ames_housing', run='baseline')
        """
        if not data.is_train:
            error_message = "Logging to MLflow is only supported for training runs."
            self.logger.error(error_message)
            raise ValueError(error_message)

        def log_params_from_config(config: dict) -> None:
            # Log top-level parameters
            for key in ["name", "description", "parameters"]:
                if key in config["pipeline"]:
                    value = config["pipeline"][key]
                    if isinstance(value, dict):
                        for key_mp, value_mp in value.items():
                            mlflow.log_param(f"pipeline.{key}.{key_mp}", value_mp)
                    else:
                        mlflow.log_param(f"pipeline.{key}", value)

            # Log step-level parameters
            for i, step in enumerate(config["pipeline"]["steps"]):
                mlflow.log_param(f"pipeline.steps_{i}.step_type", step["step_type"])
                for key, value in step.get("parameters", {}).items():
                    if isinstance(value, dict):
                        for key_mp, value_mp in value.items():
                            mlflow.log_param(
                                f"pipeline.steps_{i}.parameters.{key}.{key_mp}",
                                value_mp,
                            )
                    elif key == "model_class":
                        mlflow.log_param(f"pipeline.steps_{i}.parameters.{key}", value.__name__)
                    else:
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

        if tracking_uri:
            self.logger.info("Setting MLflow tracking URI to: %s", tracking_uri)
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment)

        if not run:
            mode_name = "train" if data.is_train else "predict"
            run = (
                f"{self.__class__.__name__}_{mode_name}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"  # noqa: DTZ005
            )
            self.logger.info("Run name not provided. Using default run name: %s", run)

        with mlflow.start_run(run_name=run):
            mlflow.set_tag("name", self.config["pipeline"]["name"])

            log_params_from_config(self.config)

            if description:
                mlflow.set_tag("mlflow.note.content", description)

            # Log prediction metrics
            if data.metrics:
                self.logger.debug("Logging prediction metrics to MLflow")
                for dataset_name, metrics in data.metrics.items():
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(f"{dataset_name}_{metric_name}", metric_value)

            # Save feature importance as a png artifact
            if data.feature_importance is not None:
                self.logger.debug("Plotting feature importance for MLflow")
                plot_feature_importance(data.feature_importance)

            if self.config:
                self.logger.debug("Logging pipeline configuration to MLflow as a JSON file")
                # convert model_class to string
                config_copy = self.config.copy()
                fit_step = next(
                    step
                    for step in config_copy["pipeline"]["steps"]
                    if issubclass(self.step_registry.get_step_class(step["step_type"]), ModelStep)
                )

                fit_step["parameters"]["model_class"] = fit_step["parameters"][
                    "model_class"
                ].__name__
                mlflow.log_dict(config_copy, "config.json")

            # save data container pickle as an artifact
            if self.save_data_path:
                self.logger.debug("Logging the data container to MLflow")
                compressed_data_path = self.save_data_path + ".zip"
                mlflow.log_artifact(compressed_data_path, artifact_path="data")

    def __str__(self) -> str:
        """Return a string representation of the pipeline with its steps."""
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
