import logging
from abc import ABC, abstractmethod

from pipeline_lib.core.data_container import DataContainer


class PipelineStep(ABC):
    """Base class for pipeline steps."""

    @abstractmethod
    def execute(self, data: DataContainer) -> DataContainer:
        """Abstract method for executing the step."""

    def init_logger(self) -> None:
        """Initialize the logger."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(f"{self.__class__.__name__} initialized")
