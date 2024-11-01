import logging
from abc import ABC, abstractmethod
from typing import Optional

from ml_garden.core.data_container import DataContainer


class PipelineStep(ABC):
    """Base class for pipeline steps."""

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config

    @abstractmethod
    def execute(self, data: DataContainer) -> DataContainer:
        """Abstract method for executing the step."""

    def init_logger(self) -> None:
        """Initialize the logger."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("%s initialized", self.__class__.__name__)
