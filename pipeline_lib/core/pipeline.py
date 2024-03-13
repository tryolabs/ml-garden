from __future__ import annotations

import json
import logging
from typing import Optional

from pipeline_lib.core.data_container import DataContainer
from pipeline_lib.core.step_registry import StepRegistry
from pipeline_lib.core.steps import PipelineStep


class Pipeline:
    """Base class for pipelines."""

    _step_registry = {}
    logger = logging.getLogger("Pipeline")
    step_registry = StepRegistry()

    def __init__(self, initial_data: Optional[DataContainer] = None):
        self.steps = []
        self.initial_data = initial_data
        self.save_path = None
        self.load_path = None

    def add_steps(self, steps: list[PipelineStep]):
        """Add steps to the pipeline."""
        self.steps.extend(steps)

    def run(self) -> DataContainer:
        """Run the pipeline on the given data."""

        data = DataContainer.from_pickle(self.load_path) if self.load_path else DataContainer()

        for i, step in enumerate(self.steps):
            Pipeline.logger.info(f"Running {step.__class__.__name__} - {i + 1} / {len(self.steps)}")
            data = step.execute(data)

        if self.save_path:
            data.save(self.save_path)

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

        for step_config in config["pipeline"]["steps"]:
            step_type = step_config["step_type"]
            parameters = step_config.get("parameters", {})

            Pipeline.logger.info(
                f"Creating step {step_type} with parameters: \n {json.dumps(parameters, indent=4)}"
            )

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
