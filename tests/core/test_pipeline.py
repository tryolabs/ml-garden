import os
from unittest.mock import MagicMock, patch

import mlflow
import pandas as pd
import pytest

from pipeline_lib import Pipeline


def test_simple_train_pipeline():
    """Test that the pipeline can be trained."""
    pipeline = Pipeline.from_json("tests/data/test.json")
    pipeline.train()

    zip_file = pipeline.save_data_path + ".zip"
    os.remove(zip_file)


def test_simple_predict_pipeline():
    """Test that the pipeline can be trained and predicted."""
    pipeline = Pipeline.from_json("tests/data/test.json")
    pipeline.train()
    pipeline.predict()

    zip_file = pipeline.save_data_path + ".zip"
    os.remove(zip_file)


def test_predict_raises_error_with_no_predict_path_and_df():
    """Test that the pipeline raises an error when no predict path is provided
    and no dataframe is provided."""
    pipeline = Pipeline.from_json("tests/data/test_without_predict_path.json")
    pipeline.train()

    # check that this raies an error
    with pytest.raises(ValueError):
        try:
            pipeline.predict()
        finally:
            zip_file = pipeline.save_data_path + ".zip"
            if os.path.exists(zip_file):
                os.remove(zip_file)


def test_predict_with_df():
    pipeline = Pipeline.from_json("tests/data/test_without_predict_path.json")
    data = pipeline.train()

    df = data.raw.drop(columns=[pipeline.target])
    pipeline.predict(df)

    zip_file = pipeline.save_data_path + ".zip"
    os.remove(zip_file)


def test_log_experiment_success(tmpdir):
    """Test that log_experiment logs the expected data to MLflow."""
    data = MagicMock()
    data.is_train = True
    data.metrics = {"train": {"accuracy": 0.9, "precision": 0.8}}
    data.feature_importance = pd.DataFrame({"feature": ["A", "B"], "importance": [0.6, 0.4]})

    class TestModel:
        pass

    pipeline = Pipeline(save_data_path=str(tmpdir), target="target")
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

    with patch("mlflow.set_tracking_uri"), patch("mlflow.set_experiment"), patch(
        "mlflow.start_run"
    ), patch("mlflow.log_param"), patch("mlflow.log_metric"), patch("mlflow.log_figure"), patch(
        "mlflow.log_dict"
    ), patch(
        "mlflow.log_artifact"
    ):
        pipeline.log_experiment(data, experiment="test_experiment")

        mlflow.set_experiment.assert_called_with("test_experiment")
        mlflow.start_run.assert_called_once()
        mlflow.log_param.assert_any_call("pipeline.name", "Test Pipeline")
        mlflow.log_param.assert_any_call("pipeline.description", "A test pipeline")
        mlflow.log_param.assert_any_call("pipeline.parameters", {"key": "value"})
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


def test_log_experiment_predict_mode_raises_error():
    """Test that log_experiment raises an error when called in predict mode."""
    data = MagicMock()
    data.is_train = False

    pipeline = Pipeline(save_data_path="", target="target")

    with pytest.raises(ValueError, match="only supported for training runs"):
        pipeline.log_experiment(data, experiment="test_experiment")
