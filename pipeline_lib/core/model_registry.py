import importlib
import logging
import pkgutil

from pipeline_lib.core.model import Model


class ModelClassNotFoundError(Exception):
    pass


class ModelRegistry:
    def __init__(self):
        self._model_registry = {}
        self.logger = logging.getLogger(__name__)

    def register_model(self, model_class: type):
        model_name = model_class.__name__
        if not issubclass(model_class, Model):
            raise ValueError(f"{model_class} must be a subclass of Model")
        self._model_registry[model_name] = model_class

    def get_model_class(self, model_name: str) -> type:
        if model_name in self._model_registry:
            return self._model_registry[model_name]
        else:
            raise ModelClassNotFoundError(f"Model class '{model_name}' not found in registry.")

    def get_all_model_classes(self) -> dict:
        return self._model_registry

    def auto_register_models_from_package(self, package_name: str):
        try:
            package = importlib.import_module(package_name)
            prefix = package.__name__ + "."
            for importer, modname, ispkg in pkgutil.walk_packages(package.__path__, prefix):
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
            self.logger.error(f"Failed to import package: {package_name}. Error: {e}")
