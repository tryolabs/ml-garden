import importlib
import logging
import pkgutil

from ml_garden.core.model import Model


class ModelClassNotFoundError(Exception):
    """Exception raised when a model class is not found in the registry."""

    pass


class ModelRegistry:
    def __init__(self) -> None:
        """
        Initialize a new ModelRegistry instance.

        Attributes
        ----------
        _model_registry : dict
            A dictionary mapping model names to model classes.
        logger : logging.Logger
            Logger for the class.
        """
        self._model_registry: dict[str, type[Model]] = {}
        self.logger = logging.getLogger(__name__)

    def register_model(self, model_class: type[Model]) -> None:
        """
        Register a model class in the registry.

        Parameters
        ----------
        model_class : type[Model]
            The model class to be registered.

        Raises
        ------
        ValueError
            If the model_class is not a subclass of Model.
        """
        model_name = model_class.__name__.lower()
        if not issubclass(model_class, Model):
            error_message = f"{model_class} must be a subclass of Model"
            self.logger.exception(error_message)
            raise TypeError(error_message)
        self._model_registry[model_name] = model_class

    def get_model_class(self, model_name: str) -> type[Model]:
        """
        Retrieve a model class from the registry.

        Parameters
        ----------
        model_name : str
            The name of the model class to retrieve.

        Returns
        -------
        type[Model]
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
            error_message = (
                f"Model class '{model_name}' not found in registry. Available models:"
                f" {list(self._model_registry.keys())}"
            )
            self.logger.exception(error_message)
            raise ModelClassNotFoundError(error_message)

    def get_all_model_classes(self) -> dict[str, type[Model]]:
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
            for _importer, modname, _ispkg in pkgutil.walk_packages(package.__path__, prefix):
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
