import pandas as pd
import pytest

from ml_garden.core import DataContainer
from ml_garden.core.constants import Task
from ml_garden.core.steps import CalculateMetricsStep


@pytest.fixture()
def data() -> DataContainer:
    """Return test data."""
    data_container = DataContainer({"is_train": True})
    data_container.train = pd.DataFrame(
        {
            "target": [1, 2, 3, 4, 5],
            "prediction": [1.1, 1.9, 3.2, 3.8, 5.1],
        }
    )
    data_container.validation = pd.DataFrame(
        {
            "target": [2, 4, 6, 8, 10],
            "prediction": [2.2, 3.8, 6.1, 7.9, 9.8],
        }
    )
    data_container.test = pd.DataFrame(
        {
            "target": [3, 6, 9, 12, 15],
            "prediction": [3.3, 5.7, 9.2, 11.8, 14.9],
        }
    )
    data_container.target = "target"
    data_container.prediction_column = "prediction"
    data_container.task = Task.REGRESSION
    return data_container


def test_train_validation_test_keys_in_metrics(data: DataContainer) -> None:
    """Test keys in the metrics dictionary."""
    step = CalculateMetricsStep()
    result = step.execute(data)

    assert isinstance(result, DataContainer)
    assert isinstance(result.metrics, dict)
    assert "train" in result.metrics
    assert "validation" in result.metrics
    assert "test" in result.metrics


def test_metrics_not_present_in_predict(data: DataContainer) -> None:
    """Test that metrics are not present in predict mode."""
    data.is_train = False
    step = CalculateMetricsStep()
    result = step.execute(data)

    assert isinstance(result, DataContainer)
    assert result.metrics is None


def test_calculate_metrics(data: DataContainer) -> None:
    """Test the _calculate_metrics method."""
    step = CalculateMetricsStep()
    metrics = step._calculate_regression_metrics(  # noqa: SLF001
        data.train["target"],
        data.train["prediction"],
    )

    assert isinstance(metrics, dict)
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "R_2" in metrics
    assert "Mean_Error" in metrics
    assert "Max_Error" in metrics
    assert "Median_Absolute_Error" in metrics
