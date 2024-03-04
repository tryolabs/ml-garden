from __future__ import annotations

import importlib
import json
import logging
import pkgutil
from typing import Optional

from pipeline_lib.core.data_container import DataContainer
from pipeline_lib.core.steps import PipelineStep


class Pipeline:
    """Base class for pipelines."""

    _step_registry = {}

    def __init__(self, initial_data: Optional[DataContainer] = None):
        self.steps = []
        if not all(isinstance(step, PipelineStep) for step in self.steps):
            raise TypeError("All steps must be instances of PipelineStep")
        self.initial_data = initial_data
        self.init_logger()

    def init_logger(self) -> None:
        """Initialize the logger."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(f"{self.__class__.__name__} initialized")

    @classmethod
    def register_step(cls, step_class):
        """Register a step class using its class name."""
        step_name = step_class.__name__
        if not issubclass(step_class, PipelineStep):
            raise ValueError(f"{step_class} must be a subclass of PipelineStep")
        cls._step_registry[step_name] = step_class

    @classmethod
    def get_step_class(cls, step_name):
        """Retrieve a step class by name."""
        if step_name in cls._step_registry:
            return cls._step_registry[step_name]
        else:
            raise ValueError(f"Step class '{step_name}' not found in registry.")

    def add_steps(self, steps: list[PipelineStep]):
        """Add steps to the pipeline."""
        self.steps.extend(steps)

    @classmethod
    def auto_register_steps_from_package(cls, package_name):
        """
        Automatically registers all step classes found within a specified package.
        """
        package = importlib.import_module(package_name)
        prefix = package.__name__ + "."
        for importer, modname, ispkg in pkgutil.walk_packages(package.__path__, prefix):
            module = importlib.import_module(modname)
            for name in dir(module):
                attribute = getattr(module, name)
                if (
                    isinstance(attribute, type)
                    and issubclass(attribute, PipelineStep)
                    and attribute is not PipelineStep
                ):
                    cls.register_step(attribute)

    def run(self) -> DataContainer:
        """Run the pipeline on the given data."""

        data = DataContainer()

        for i, step in enumerate(self.steps):
            self.logger.info(f"Running {step.__class__.__name__} - {i + 1} / {len(self.steps)}")
            data = step.execute(data)
        return data

    @classmethod
    def from_json(cls, path: str) -> Pipeline:
        """Load a pipeline from a JSON file."""
        # check file is a json file
        if not path.endswith(".json"):
            raise ValueError(f"File {path} is not a JSON file")

        with open(path, "r") as config_file:
            config = json.load(config_file)

        pipeline = Pipeline()  # Assuming you have a default or base Pipeline class
        steps = []

        for step_config in config["pipeline"]["steps"]:
            step_type = step_config["step_type"]
            parameters = step_config.get("parameters", {})

            pipeline.logger.info(
                f"Creating step {step_type} with parameters: \n {json.dumps(parameters, indent=4)}"
            )

            step_class = Pipeline.get_step_class(step_type)
            step = step_class(config=parameters)
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
