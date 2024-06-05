from unittest.mock import MagicMock, patch

import pytest

from ml_garden.core.model import Model
from ml_garden.core.model_registry import ModelClassNotFoundError, ModelRegistry


class DummyModel(Model):
    pass


class AnotherDummyModel(Model):
    pass


def test_register_model():
    registry = ModelRegistry()
    registry.register_model(DummyModel)
    assert DummyModel in registry.get_all_model_classes().values()


def test_register_non_model_class():
    registry = ModelRegistry()
    with pytest.raises(ValueError):
        registry.register_model(int)


def test_get_model_class():
    registry = ModelRegistry()
    registry.register_model(DummyModel)
    assert registry.get_model_class("DummyModel") == DummyModel


def test_get_nonexistent_model_class():
    registry = ModelRegistry()
    with pytest.raises(ModelClassNotFoundError):
        registry.get_model_class("NonexistentModel")


def test_get_all_model_classes():
    registry = ModelRegistry()
    registry.register_model(DummyModel)
    registry.register_model(AnotherDummyModel)
    all_models = registry.get_all_model_classes()
    assert len(all_models) == 2
    assert DummyModel in all_models.values()
    assert AnotherDummyModel in all_models.values()


@patch("ml_garden.core.model_registry.pkgutil.walk_packages")
@patch("ml_garden.core.model_registry.importlib.import_module")
def test_auto_register_models_from_package(mock_import_module, mock_walk_packages):
    mock_package = MagicMock()
    mock_package.__name__ = "package"
    mock_package.__path__ = ["package/path"]

    mock_module1 = MagicMock()
    mock_module1.__name__ = "package.module1"
    mock_module1.Model1 = DummyModel
    mock_module1.Model2 = AnotherDummyModel

    mock_module2 = MagicMock()
    mock_module2.__name__ = "package.module2"
    mock_module2.Model3 = DummyModel

    mock_import_module.side_effect = [mock_package, mock_module1, mock_module2]

    mock_walk_packages.return_value = [
        (None, "package.module1", False),
        (None, "package.module2", False),
    ]

    registry = ModelRegistry()
    registry.auto_register_models_from_package("package")

    assert len(registry.get_all_model_classes()) == 2
    assert DummyModel in registry.get_all_model_classes().values()
    assert AnotherDummyModel in registry.get_all_model_classes().values()


@patch("ml_garden.core.model_registry.importlib.import_module")
def test_auto_register_models_import_error(mock_import_module):
    mock_import_module.side_effect = ImportError

    registry = ModelRegistry()
    registry.auto_register_models_from_package("invalid_package")

    assert len(registry.get_all_model_classes()) == 0
