from unittest.mock import MagicMock, patch

import pytest

from ml_garden.core.step_registry import StepClassNotFoundError, StepRegistry
from ml_garden.core.steps.base import PipelineStep


class DummyStep(PipelineStep):
    pass


class AnotherDummyStep(PipelineStep):
    pass


def test_register_step():
    registry = StepRegistry()
    registry.register_step(DummyStep)
    assert DummyStep in registry.get_all_step_classes().values()


def test_register_non_step_class():
    registry = StepRegistry()
    with pytest.raises(ValueError):
        registry.register_step(int)


def test_get_step_class():
    registry = StepRegistry()
    registry.register_step(DummyStep)
    assert registry.get_step_class("DummyStep") == DummyStep


def test_get_nonexistent_step_class():
    registry = StepRegistry()
    with pytest.raises(StepClassNotFoundError):
        registry.get_step_class("NonexistentStep")


def test_get_all_step_classes():
    registry = StepRegistry()
    registry.register_step(DummyStep)
    registry.register_step(AnotherDummyStep)
    all_steps = registry.get_all_step_classes()
    assert len(all_steps) == 2
    assert DummyStep in all_steps.values()
    assert AnotherDummyStep in all_steps.values()


@patch("ml_garden.core.step_registry.pkgutil.walk_packages")
@patch("ml_garden.core.step_registry.importlib.import_module")
def test_auto_register_steps_from_package(mock_import_module, mock_walk_packages):
    mock_package = MagicMock()
    mock_package.__name__ = "package"
    mock_package.__path__ = ["package/path"]

    mock_module1 = MagicMock()
    mock_module1.__name__ = "package.module1"
    mock_module1.Step1 = DummyStep
    mock_module1.Step2 = AnotherDummyStep

    mock_module2 = MagicMock()
    mock_module2.__name__ = "package.module2"
    mock_module2.Step3 = DummyStep

    mock_import_module.side_effect = [mock_package, mock_module1, mock_module2]

    mock_walk_packages.return_value = [
        (None, "package.module1", False),
        (None, "package.module2", False),
    ]

    registry = StepRegistry()
    registry.auto_register_steps_from_package("package")

    assert len(registry.get_all_step_classes()) == 2
    assert DummyStep in registry.get_all_step_classes().values()
    assert AnotherDummyStep in registry.get_all_step_classes().values()


@patch("ml_garden.core.step_registry.importlib.import_module")
def test_auto_register_steps_import_error(mock_import_module):
    mock_import_module.side_effect = ImportError

    registry = StepRegistry()
    registry.auto_register_steps_from_package("invalid_package")

    assert len(registry.get_all_step_classes()) == 0


@patch("ml_garden.core.step_registry.os.listdir")
@patch("ml_garden.core.step_registry.importlib.util.spec_from_file_location")
@patch("ml_garden.core.step_registry.importlib.util.module_from_spec")
def test_load_and_register_custom_steps(
    mock_module_from_spec, mock_spec_from_file_location, mock_listdir
):
    mock_listdir.return_value = ["custom_step.py"]

    mock_spec = MagicMock()
    mock_spec_from_file_location.return_value = mock_spec

    mock_module = MagicMock()
    mock_module_from_spec.return_value = mock_module
    mock_module.CustomStep = DummyStep

    registry = StepRegistry()
    registry.load_and_register_custom_steps("custom_steps_path")

    assert len(registry.get_all_step_classes()) == 1
    assert DummyStep in registry.get_all_step_classes().values()


@patch("ml_garden.core.step_registry.os.listdir")
@patch("ml_garden.core.step_registry.importlib.util.spec_from_file_location")
@patch("ml_garden.core.step_registry.importlib.util.module_from_spec")
def test_load_and_register_custom_steps_exception(
    mock_module_from_spec, mock_spec_from_file_location, mock_listdir
):
    mock_listdir.return_value = ["custom_step.py"]

    mock_spec = MagicMock()
    mock_spec_from_file_location.return_value = mock_spec

    mock_module = MagicMock()
    mock_module_from_spec.return_value = mock_module
    mock_module.CustomStep = DummyStep

    mock_spec.loader.exec_module.side_effect = Exception("Test Exception")

    registry = StepRegistry()
    registry.load_and_register_custom_steps("custom_steps_path")

    assert len(registry.get_all_step_classes()) == 0
