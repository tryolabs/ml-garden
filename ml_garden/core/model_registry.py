import importlib
import logging
import pkgutil
from typing import Dict, Type

from ml_garden.core.model import Model
from ml_garden.core.steps.fit_model_autogluon import AutoGluonModel


class ModelClassNotFoundError(Exception):
    """Exception raised when a model class is not found in the registry."""

    pass


class ModelRegistry:
    def __init__(self):
        """
        Initialize a new ModelRegistry instance.

        Attributes
        ----------
        _model_registry : dict
            A dictionary mapping model names to model classes.
        logger : logging.Logger
            Logger for the class.
        """
        self._model_registry: Dict[str, Type[Model]] = {}
        self.logger = logging.getLogger(__name__)
        self.register_model(AutoGluonModel)

    def register_model(self, model_class: Type[Model]) -> None:
        """
        Register a model class in the registry.

        Parameters
        ----------
        model_class : Type[Model]
            The model class to be registered.

        Raises
        ------
        ValueError
            If the model_class is not a subclass of Model.
        """
        model_name = model_class.__name__.lower()
        if not issubclass(model_class, Model):
            raise ValueError(f"{model_class} must be a subclass of Model")
        self._model_registry[model_name] = model_class

    def get_model_class(self, model_name: str) -> Type[Model]:
        """
        Retrieve a model class from the registry.

        Parameters
        ----------
        model_name : str
            The name of the model class to retrieve.

        Returns
        -------
        Type[Model]
            The model class.

        Raises
        ------
        ModelClassNotFoundError
            If the model class is not found in the registry.
        """
        model_name = model_name.lower()
        if model_name in self._model_registry:
            return self._model_registry[model_name]
        else:
            raise ModelClassNotFoundError(
                f"Model class '{model_name}' not found in registry. Available models:"
                f" {list(self._model_registry.keys())}"
            )

    def get_all_model_classes(self) -> Dict[str, Type[Model]]:
        """
        Get all registered model classes.

        Returns
        -------
        dict
            A dictionary of all registered model classes.
        """
        return self._model_registry

    def auto_register_models_from_package(self, package_name: str) -> None:
        """
        Automatically register all model classes from a given package.

        Parameters
        ----------
        package_name : str
            The name of the package to search for model classes.

        Raises
        ------
        ImportError
            If the package cannot be imported.
        """
        try:
            package = importlib.import_module(package_name)
            prefix = package.__name__ + "."
            for importer, modname, ispkg in pkgutil.walk_packages(
                package.__path__, prefix
            ):
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
