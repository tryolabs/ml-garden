import importlib
import logging
import pkgutil

from ml_garden.core.model import Model


class ModelClassNotFoundError(Exception):
    pass


class ModelRegistry:
    def __init__(self) -> None:
        self._model_registry = {}
        self.logger = logging.getLogger(__name__)

    def register_model(self, model_class: type) -> None:
        model_name = model_class.__name__
        if not issubclass(model_class, Model):
            error_message = f"{model_class} must be a subclass of Model"
            self.logger.exception(error_message)
            raise TypeError(error_message)
        self._model_registry[model_name] = model_class

    def get_model_class(self, model_name: str) -> type:
        if model_name in self._model_registry:
            return self._model_registry[model_name]
        else:
            error_message = (
                f"Model class '{model_name}' not found in registry. Available models:"
                f" {list(self._model_registry.keys())}"
            )
            self.logger.exception(error_message)
            raise ModelClassNotFoundError(error_message)

    def get_all_model_classes(self) -> dict:
        return self._model_registry

    def auto_register_models_from_package(self, package_name: str) -> None:
        try:
            package = importlib.import_module(package_name)
            prefix = package.__name__ + "."
            for _, modname, _ in pkgutil.walk_packages(package.__path__, prefix):
                module = importlib.import_module(modname)
                for name in dir(module):
                    attribute = getattr(module, name)
                    if (
                        isinstance(attribute, type)
                        and issubclass(attribute, Model)
                        and attribute is not Model
                    ):
                        self.register_model(attribute)
        except ImportError as e:
            error_message = f"Failed to import package: {package_name}. Error: {e}"
            self.logger.exception(error_message)
            raise ImportError(error_message) from e
