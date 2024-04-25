from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Optional

from pipeline_lib.core.data_container import DataContainer
from pipeline_lib.core.model_registry import ModelRegistry
from pipeline_lib.core.step_registry import StepRegistry
from pipeline_lib.core.steps import PipelineStep


class Pipeline:
    """Base class for pipelines."""

    _step_registry = {}
    logger = logging.getLogger("Pipeline")
    step_registry = StepRegistry()
    model_registry = ModelRegistry()

    KEYS_TO_SAVE = [
        "target",
        "model",
        "encoder",
        "_drop_columns",
        "target",
        "prediction_column",
        "_generate_step_dtypes",
        "explainer",
        "columns_to_ignore_for_training",
    ]

    def __init__(self, initial_data: Optional[DataContainer] = None):
        self.steps = []
        self.initial_data = initial_data
        self.config = {}
        self.save_data_path = None

    def add_steps(self, steps: list[PipelineStep]):
        """Add steps to the pipeline."""
        self.steps.extend(steps)

    def run(self, is_train: bool, save: bool = True) -> DataContainer:
        """Run the pipeline on the given data."""

        if "parameters" not in self.config["pipeline"]:
            raise ValueError("Missing pipeline parameters section in the config file.")

        if "save_data_path" not in self.config["pipeline"]["parameters"]:
            raise ValueError(
                "A path for saving the data must be provided. Use the `save_data_path` attribute "
                'of the pipeline parameters" section in the config.'
            )

        if "target" not in self.config["pipeline"]["parameters"]:
            raise ValueError(
                "A target column must be provided. Use the `target` attribute of the pipeline"
                ' "parameters" section in the config.'
            )

        data = DataContainer()

        self.save_data_path = self.config["pipeline"]["parameters"]["save_data_path"]
        data.target = self.config["pipeline"]["parameters"]["target"]
        data.prediction_column = f"{data.target}_prediction"
        data.columns_to_ignore_for_training = self.config["pipeline"]["parameters"].get(
            "columns_to_ignore_for_training", []
        )

        if is_train:
            steps_to_run = [step for step in self.steps if step.used_for_training]
            self.logger.info("Training the pipeline")
        else:
            data = DataContainer.from_pickle(self.save_data_path)
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
    def from_json(cls, path: str) -> Pipeline:
        """Load a pipeline from a JSON file."""
        # check file is a json file
        if not path.endswith(".json"):
            raise ValueError(f"File {path} is not a JSON file")

        with open(path, "r") as config_file:
            config = json.load(config_file)

        custom_steps_path = config.get("custom_steps_path")
        if custom_steps_path:
            cls.step_registry.load_and_register_custom_steps(custom_steps_path)

        pipeline = Pipeline()

        pipeline.config = config
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

    def save_run(
        self,
        data: DataContainer,
        parent_folder: str = "runs",
        logs: Optional[logging.LogRecord] = None,
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
                json.dump(data.metrics, f, indent=4)

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
