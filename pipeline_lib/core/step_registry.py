import importlib
import logging
import os
import pkgutil

from pipeline_lib.core.steps import PipelineStep


class StepClassNotFoundError(Exception):
    pass


class StepRegistry:
    """A helper class for managing the registry of pipeline steps."""

    def __init__(self):
        self._step_registry = {}
        self.logger = logging.getLogger(__name__)

    def register_step(self, step_class: type):
        """Register a step class using its class name."""
        step_name = step_class.__name__
        if not issubclass(step_class, PipelineStep):
            raise ValueError(f"{step_class} must be a subclass of PipelineStep")
        self._step_registry[step_name] = step_class

    def get_step_class(self, step_name: str) -> type:
        """Retrieve a step class by name."""
        if step_name in self._step_registry:
            return self._step_registry[step_name]
        else:
            raise StepClassNotFoundError(f"Step class '{step_name}' not found in registry.")

    def get_all_step_classes(self) -> dict:
        """Retrieve all registered step classes."""
        return self._step_registry

    def auto_register_steps_from_package(self, package_name: str):
        """
        Automatically registers all step classes found within a specified package.
        """
        try:
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
                        self.register_step(attribute)
        except ImportError as e:
            self.logger.error(f"Failed to import package: {package_name}. Error: {e}")

    def load_and_register_custom_steps(self, custom_steps_path: str) -> None:
        """
        Dynamically loads and registers step classes found in the specified directory.
        """
        self.logger.debug(f"Loading custom steps from: {custom_steps_path}")
        for filename in os.listdir(custom_steps_path):
            if filename.endswith(".py") and not filename.startswith("__"):
                filepath = os.path.join(custom_steps_path, filename)
                module_name = os.path.splitext(filename)[0]
                spec = importlib.util.spec_from_file_location(module_name, filepath)
                module = importlib.util.module_from_spec(spec)

                try:
                    spec.loader.exec_module(module)
                    self.logger.debug(f"Successfully loaded module: {module_name}")

                    for attribute_name in dir(module):
                        attribute = getattr(module, attribute_name)
                        if (
                            isinstance(attribute, type)
                            and issubclass(attribute, PipelineStep)
                            and attribute is not PipelineStep
                        ):
                            self.register_step(attribute)
                            self.logger.debug(f"Registered step class: {attribute_name}")
                except Exception as e:
                    self.logger.error(f"Failed to load module: {module_name}. Error: {e}")
