from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

from pipeline_lib.core.data_container import DataContainer
from pipeline_lib.core.steps import PipelineStep


class Pipeline(ABC):
    """Base class for pipelines."""

    def __init__(self, initial_data: Optional[DataContainer] = None):
        self.steps = self.define_steps()
        if not all(isinstance(step, PipelineStep) for step in self.steps):
            raise TypeError("All steps must be instances of PipelineStep")
        self.initial_data = initial_data

    def run(self, data: Optional[DataContainer] = None) -> DataContainer:
        """Run the pipeline on the given data."""
        if data is None:
            if self.initial_data is None:
                raise ValueError("No data given and no initial data set")
            self.logger.debug("No data given, using initial data")
            data = self.initial_data

        for i, step in enumerate(self.steps):
            self.logger.info(f"Running {step.__class__.__name__} - {i + 1} / {len(self.steps)}")
            data = step.execute(data)
        return data

    @abstractmethod
    def define_steps(self) -> list[PipelineStep]:
        """
        Subclasses should implement this method to define their specific steps.
        """

    def init_logger(self) -> None:
        """Initialize the logger."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(f"{self.__class__.__name__} initialized")

    def __str__(self) -> str:
        step_names = [f"{i + 1}. {step.__class__.__name__}" for i, step in enumerate(self.steps)]
        return f"{self.__class__.__name__} with steps:\n" + "\n".join(step_names)

    def __repr__(self) -> str:
        """Return an unambiguous string representation of the pipeline."""
        step_names = [f"{step.__class__.__name__}()" for step in self.steps]
        return f"{self.__class__.__name__}({', '.join(step_names)})"
