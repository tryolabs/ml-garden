from pathlib import Path
from unittest.mock import MagicMock, patch

import mlflow
import pandas as pd
import pytest

from ml_garden import Pipeline
from ml_garden.core.constants import Task


def test_simple_train_pipeline() -> None:
    """Test that the pipeline can be trained."""
    pipeline = Pipeline.from_json("tests/data/test.json")
    pipeline.train()

    zip_file = Path(pipeline.save_data_path + ".zip")
    zip_file.unlink()


def test_simple_train_pipeline_classification() -> None:
    """Test that the pipeline can be trained with classification."""
    pipeline = Pipeline.from_json("tests/data/test_classification.json")
    pipeline.train()

    zip_file = Path(pipeline.save_data_path + ".zip")
    zip_file.unlink()


def test_simple_predict_pipeline() -> None:
    """Test that the pipeline can be trained and predicted."""
    pipeline = Pipeline.from_json("tests/data/test.json")
    pipeline.train()
    pipeline.predict()

    zip_file = Path(pipeline.save_data_path + ".zip")
    zip_file.unlink()


def test_simple_predict_classification_pipeline() -> None:
    """Test that the pipeline can be trained and predicted."""
    pipeline = Pipeline.from_json("tests/data/test_classification.json")
    pipeline.train()
    pipeline.predict()

    zip_file = Path(pipeline.save_data_path + ".zip")
    zip_file.unlink()


def test_predict_raises_error_with_no_predict_path_and_df() -> None:
    """Test that the pipeline raises an error when no predict path is provided.

    Also no dataframe is provided.
    """
    pipeline = Pipeline.from_json("tests/data/test_without_predict_path.json")
    pipeline.train()

    with pytest.raises(
        ValueError,
        match="predict_path was not set in the configuration file, and no DataFrame was provided"
        " for prediction",
    ):
        pipeline.predict()

    zip_file = Path(pipeline.save_data_path + ".zip")
    if zip_file.exists():
        zip_file.unlink()


def test_predict_with_df() -> None:
    pipeline = Pipeline.from_json("tests/data/test_without_predict_path.json")
    data = pipeline.train()

    df = data.raw.drop(columns=[pipeline.target])
    pipeline.predict(df)

    zip_file = Path(pipeline.save_data_path + ".zip")
    zip_file.unlink()


def test_log_experiment_success(tmpdir: Path) -> None:
    """Test that log_experiment logs the expected data to MLflow."""
    data = MagicMock()
    data.is_train = True
    data.metrics = {"train": {"accuracy": 0.9, "precision": 0.8}}
    data.feature_importance = pd.DataFrame({"feature": ["A", "B"], "importance": [0.6, 0.4]})

    class TestModel:
        pass

    pipeline = Pipeline(save_data_path=str(tmpdir), target="target", task=Task.REGRESSION)
    pipeline.config = {
        "pipeline": {
            "name": "Test Pipeline",
            "description": "A test pipeline",
            "parameters": {"key": "value"},
            "steps": [
                {"step_type": "TestStep", "parameters": {"param": "value"}},
                {"step_type": "ModelStep", "parameters": {"model_class": TestModel}},
            ],
        }
    }

    with (
        patch("mlflow.set_tracking_uri"),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run"),
        patch("mlflow.log_param"),
        patch("mlflow.log_metric"),
        patch("mlflow.log_figure"),
        patch("mlflow.log_dict"),
        patch("mlflow.log_artifact"),
    ):
        pipeline.log_experiment(data, experiment="test_experiment")

        mlflow.set_experiment.assert_called_with("test_experiment")
        mlflow.start_run.assert_called_once()
        mlflow.log_param.assert_any_call("pipeline.name", "Test Pipeline")
        mlflow.log_param.assert_any_call("pipeline.description", "A test pipeline")
        mlflow.log_param.assert_any_call("pipeline.parameters.key", "value")
        mlflow.log_param.assert_any_call("pipeline.steps_0.step_type", "TestStep")
        mlflow.log_param.assert_any_call("pipeline.steps_0.parameters.param", "value")
        mlflow.log_param.assert_any_call("pipeline.steps_1.step_type", "ModelStep")
        mlflow.log_param.assert_any_call(
            "pipeline.steps_1.parameters.model_class", TestModel.__name__
        )
        mlflow.log_metric.assert_any_call("train_accuracy", 0.9)
        mlflow.log_metric.assert_any_call("train_precision", 0.8)
        mlflow.log_figure.assert_called_once()
        mlflow.log_dict.assert_called_once_with(pipeline.config, "config.json")
        mlflow.log_artifact.assert_called_once()


def test_log_experiment_predict_mode_raises_error() -> None:
    """Test that log_experiment raises an error when called in predict mode."""
    data = MagicMock()
    data.is_train = False

    pipeline = Pipeline(save_data_path="", target="target", task=Task.REGRESSION)

    with pytest.raises(ValueError, match="only supported for training runs"):
        pipeline.log_experiment(data, experiment="test_experiment")


def test_check_ames_housing_performance() -> None:
    """Test that the testing mae is less than 16000 for Ames Housing problem."""
    pipeline = Pipeline.from_json("tests/data/ames_housing.json")
    data = pipeline.train()
    metrics = data.metrics
    test_mae = float(metrics["test"]["MAE"])

    expected_mae = 16000.0

    assert test_mae < expected_mae

    zip_file = Path(pipeline.save_data_path + ".zip")
    zip_file.unlink()
