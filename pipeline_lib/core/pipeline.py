from __future__ import annotations

import json
import logging
from typing import Optional

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

    def __init__(self, initial_data: Optional[DataContainer] = None):
        self.steps = []
        self.initial_data = initial_data
        self.save_path = None
        self.load_path = None
        self.model_path = None

    def add_steps(self, steps: list[PipelineStep]):
        """Add steps to the pipeline."""
        self.steps.extend(steps)

    def run(self, is_train: bool) -> DataContainer:
        """Run the pipeline on the given data."""

        data = DataContainer.from_pickle(self.load_path) if self.load_path else DataContainer()
        data.is_train = is_train

        if is_train:
            steps_to_run = [step for step in self.steps if step.used_for_training]
        else:
            steps_to_run = [step for step in self.steps if step.used_for_prediction]

        for i, step in enumerate(steps_to_run):
            Pipeline.logger.info(
                f"Running {step.__class__.__name__} - {i + 1} / {len(steps_to_run)}"
            )
            data = step.execute(data)

        if self.save_path:
            data.save(self.save_path)

        return data

    def train(self) -> DataContainer:
        """Run the pipeline on the given data."""
        self.logger.info("Training the pipeline")
        return self.run(is_train=True)

    def predict(self) -> DataContainer:
        """Run the pipeline on the given data."""
        self.logger.info("Predicting with the pipeline")
        data = self.run(is_train=False)
        data.predictions = data.model.predict(data.flow)
        self.logger.info("Predictions:")
        self.logger.info(data.predictions)
        return data

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

        pipeline.load_path = config.get("load_path")
        pipeline.save_path = config.get("save_path")

        steps = []

        model_path = None
        drop_columns = None
        target = None

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
                model_path = parameters.get("save_path")
                drop_columns = parameters.get("drop_columns")
                target = parameters.get("target")

            # if step type is prediction, add model path
            if step_type == "PredictStep":
                parameters["load_path"] = model_path
                parameters["drop_columns"] = drop_columns
                parameters["target"] = target

            step_class = cls.step_registry.get_step_class(step_type)
            step = step_class(**parameters)
            steps.append(step)

        pipeline.add_steps(steps)
        return pipeline

    def __str__(self) -> str:
        step_names = [f"{i + 1}. {step.__class__.__name__}" for i, step in enumerate(self.steps)]
        return f"{self.__class__.__name__} with steps:\n" + "\n".join(step_names)

    def __repr__(self) -> str:
        """Return an unambiguous string representation of the pipeline."""
        step_names = [f"{step.__class__.__name__}()" for step in self.steps]
        return f"{self.__class__.__name__}({', '.join(step_names)})"
