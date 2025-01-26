from unittest import mock

import pandas as pd
import pytest

from ml_garden.core import DataContainer
from ml_garden.core.constants import Task
from ml_garden.core.model import Model
from ml_garden.core.steps.fit_model import ModelStep, OptunaOptimizer


@pytest.fixture()
def mock_model_class() -> mock.MagicMock:
    mock_model = mock.MagicMock(spec=Model)
    mock_model.__name__ = "MockModel"
    return mock_model


@pytest.fixture()
def data_container() -> DataContainer:
    # Create a sample DataContainer for testing
    data = {
        "X_train": pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]}),
        "y_train": pd.Series([0, 1, 1]),
        "X_validation": pd.DataFrame({"feature1": [7, 8], "feature2": [9, 10]}),
        "y_validation": pd.Series([0, 1]),
        "X_test": pd.DataFrame({"feature1": [11, 12], "feature2": [13, 14]}),
        "y_test": pd.Series([1, 0]),
        "train": pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6], "target": [0, 1, 1]}),
        "validation": pd.DataFrame({"feature1": [7, 8], "feature2": [9, 10], "target": [0, 1]}),
        "test": pd.DataFrame({"feature1": [11, 12], "feature2": [13, 14], "target": [1, 0]}),
        "prediction_column": "predicted",
        "task": Task.REGRESSION,
    }
    return DataContainer(data)


def test_train(mock_model_class: mock.MagicMock, data_container: DataContainer) -> None:
    model_step = ModelStep(model_class=mock_model_class)

    # Set the return values of the predict method for each dataset
    mock_train_predictions = pd.Series([0, 1, 1], name=data_container.prediction_column)
    mock_validation_predictions = pd.Series([0, 1], name=data_container.prediction_column)
    mock_test_predictions = pd.Series([1, 0], name=data_container.prediction_column)

    def mock_predict(X: pd.DataFrame) -> pd.Series:  # noqa: N803
        if X.equals(data_container.X_train):
            return mock_train_predictions
        elif X.equals(data_container.X_validation):
            return mock_validation_predictions
        elif X.equals(data_container.X_test):
            return mock_test_predictions

    mock_model_class.return_value.predict.side_effect = mock_predict

    result = model_step.train(data_container)

    mock_model_class.assert_called_once_with()
    mock_model_class.return_value.fit.assert_called_once_with(
        data_container.X_train,
        data_container.y_train,
        eval_set=[(data_container.X_validation, data_container.y_validation)],
        verbose=True,
    )
    assert result.model == mock_model_class.return_value
    assert result.train[result.prediction_column].equals(mock_train_predictions)
    assert result.validation[result.prediction_column].equals(mock_validation_predictions)
    assert result.test[result.prediction_column].equals(mock_test_predictions)


def test_predict(mock_model_class: mock.MagicMock, data_container: DataContainer) -> None:
    # Set the is_train attribute to False to simulate a prediction scenario
    data_container.is_train = False

    # Define the mock prediction data
    mock_prediction_data = pd.DataFrame({"feature1": [15, 16], "feature2": [17, 18]})
    mock_predictions = pd.Series([0, 1], name=data_container.prediction_column)

    # Set up the data container for prediction
    data_container.X_prediction = mock_prediction_data
    data_container.model = mock_model_class()

    # Initialize data.flow as a DataFrame with the same columns as X_prediction
    data_container.flow = mock_prediction_data.copy()

    # Define the mock predict method to return the mock predictions
    data_container.model.predict.return_value = mock_predictions

    # Create an instance of ModelStep
    model_step = ModelStep(model_class=mock_model_class)

    # Execute the predict method
    result = model_step.predict(data_container)

    # Assert that the predict method was called once with the correct data
    data_container.model.predict.assert_called_once_with(data_container.X_prediction)

    # Assert that the predictions in the result match the mock predictions
    assert result.flow[data_container.prediction_column].equals(mock_predictions)
    assert result.predictions.equals(mock_predictions)


def test_optuna_optimizer(mock_model_class: mock.MagicMock, data_container: DataContainer) -> None:
    optuna_params = {
        "objective_metric": "mae",
        "trials": 5,
        "search_space": {
            "param1": {"type": "float", "args": [0.1, 0.9]},
            "param2": {"type": "int", "args": [1, 10]},
        },
    }
    initial_model_parameters = {"param1": 0.5, "param2": 5}
    optimized_model_parameters = {"param1": 0.7, "param2": 3}

    model_step = ModelStep(
        model_class=mock_model_class,
        model_parameters=initial_model_parameters,
        optuna_params=optuna_params,
    )

    with mock.patch.object(
        OptunaOptimizer, "optimize", return_value=optimized_model_parameters
    ) as mock_optimize:
        result = model_step.train(data_container)

        # Assert the call to optimize was made with the correct parameters
        mock_optimize.assert_called_once_with(
            data_container.X_train,
            data_container.y_train,
            data_container.X_validation,
            data_container.y_validation,
            mock_model_class,
            initial_model_parameters,
            data_container.task,
        )

    # Assert the model is instantiated with optimized parameters
    mock_model_class.assert_called_once_with(**optimized_model_parameters)

    # Assert the fit method is called with the correct data
    mock_model_class.return_value.fit.assert_called_once_with(
        data_container.X_train,
        data_container.y_train,
        eval_set=[(data_container.X_validation, data_container.y_validation)],
        verbose=True,
    )

    # Assert the result model is the mock model instance
    assert result.model == mock_model_class.return_value


def test_train_missing_data(
    mock_model_class: mock.MagicMock, data_container: DataContainer
) -> None:
    model_step = ModelStep(model_class=mock_model_class)
    data_container.X_train = None
    data_container.y_train = None

    with pytest.raises(ValueError, match="Encoded train data not found"):
        model_step.train(data_container)


def test_optuna_optimizer_unsupported_metric(
    mock_model_class: mock.MagicMock, data_container: DataContainer
) -> None:
    optuna_params = {
        "objective_metric": "unsupported_metric",
        "trials": 5,
        "search_space": {
            "param1": {"type": "float", "args": [0.1, 0.9]},
            "param2": {"type": "int", "args": [1, 10]},
        },
    }
    model_step = ModelStep(
        model_class=mock_model_class,
        optuna_params=optuna_params,
    )

    with pytest.raises(ValueError, match="Unsupported objective metric"):
        model_step.train(data_container)


def test_execute(mock_model_class: mock.MagicMock, data_container: DataContainer) -> None:
    model_step = ModelStep(model_class=mock_model_class)

    with mock.patch.object(model_step, "train") as mock_train:
        data_container.is_train = True
        model_step.execute(data_container)
        mock_train.assert_called_once_with(data_container)

    with mock.patch.object(model_step, "predict") as mock_predict:
        data_container.is_train = False
        model_step.execute(data_container)
        mock_predict.assert_called_once_with(data_container)
