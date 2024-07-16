import numpy as np
import pandas as pd
import pytest

from ml_garden.core import DataContainer
from ml_garden.core.steps.calculate_features import (
    CalculateFeaturesStep,
    UnsupportedFeatureError,
)


@pytest.fixture
def input_data() -> pd.DataFrame:
    # Data as a dictionary
    data = {
        "creation_date": np.array(
            [
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
                "2023-01-04",
                "2023-01-05",
                "2023-11-01",
                "2024-02-28",
                "2024-03-28",
            ],
            dtype=str,
        ),
        "deletion_date": np.array(
            [
                "2024-11-05",
                "2024-11-07",
                "2024-12-20",
                "2024-12-31",
                "2025-02-18",
                "2025-02-24",
                "2025-03-12",
                "2026-10-13",
            ],
            dtype=str,
        ),
        "incorrect_date": np.array(
            [
                "202A-01-01",
                "202B-01-02",
                "202C-01-03",
                "202D-01-04",
                "202E-01-05",
                "202F-11-01",
                "202G-02-28",
                "202H-03-28",
            ],
            dtype=str,
        ),
    }

    # Create the DataFrame
    return pd.DataFrame(data)


@pytest.fixture
def data(input_data: pd.DataFrame) -> DataContainer:
    data = DataContainer({"is_train": True})
    data.columns_to_ignore_for_training = []
    data.X_train = input_data
    return data


def test_skipping_with_no_parameters(data: DataContainer):
    """Test to check if the step is skipped when no parameters are provided."""
    calculate_features_step = CalculateFeaturesStep()
    result = calculate_features_step.execute(data)

    assert isinstance(result, DataContainer)
    assert result.X_train.equals(data.X_train)


def test_feature_names(data: DataContainer):
    """Test to check correct naming of feature columns"""
    datetime_columns = ["creation_date", "deletion_date"]
    features = ["year", "month", "day", "hour", "minute", "second", "weekday", "dayofyear"]

    calculate_features_step = CalculateFeaturesStep(
        datetime_columns=datetime_columns,
        features=features,
    )
    result = calculate_features_step.execute(data)

    assert isinstance(result, DataContainer)
    assert "creation_date_year" in result.X_train.columns
    assert "creation_date_month" in result.X_train.columns
    assert "creation_date_day" in result.X_train.columns
    assert "creation_date_hour" in result.X_train.columns
    assert "creation_date_minute" in result.X_train.columns
    assert "creation_date_second" in result.X_train.columns
    assert "creation_date_weekday" in result.X_train.columns
    assert "creation_date_dayofyear" in result.X_train.columns
    assert "deletion_date_year" in result.X_train.columns
    assert "deletion_date_month" in result.X_train.columns
    assert "deletion_date_day" in result.X_train.columns
    assert "deletion_date_hour" in result.X_train.columns
    assert "deletion_date_minute" in result.X_train.columns
    assert "deletion_date_second" in result.X_train.columns
    assert "deletion_date_weekday" in result.X_train.columns
    assert "deletion_date_dayofyear" in result.X_train.columns


def test_date_columns_are_ignored_for_training(data: DataContainer):
    """Test to check if the date columns are ignored for training."""
    datetime_columns = ["creation_date", "deletion_date"]
    features = ["year", "month", "day"]

    calculate_features_step = CalculateFeaturesStep(
        datetime_columns=datetime_columns,
        features=features,
    )
    result = calculate_features_step.execute(data)

    assert isinstance(result, DataContainer)
    assert "creation_date" not in result.X_train.columns
    assert "deletion_date" not in result.X_train.columns


def test_output_dtypes(data: DataContainer):
    """Test to check the output data types."""
    datetime_columns = ["creation_date"]
    features = ["year", "month", "day", "hour", "minute", "second", "weekday", "dayofyear"]

    calculate_features_step = CalculateFeaturesStep(
        datetime_columns=datetime_columns,
        features=features,
    )
    result = calculate_features_step.execute(data)

    assert isinstance(result, DataContainer)
    assert result.X_train["creation_date_year"].dtype == np.dtype("uint16")
    assert result.X_train["creation_date_month"].dtype == np.dtype("uint8")
    assert result.X_train["creation_date_day"].dtype == np.dtype("uint8")
    assert result.X_train["creation_date_hour"].dtype == np.dtype("uint8")
    assert result.X_train["creation_date_minute"].dtype == np.dtype("uint8")
    assert result.X_train["creation_date_second"].dtype == np.dtype("uint8")
    assert result.X_train["creation_date_weekday"].dtype == np.dtype("uint8")
    assert result.X_train["creation_date_dayofyear"].dtype == np.dtype("uint16")


def test_output_values(data: DataContainer):
    """Test to check the output values."""
    datetime_columns = ["creation_date"]
    features = ["year", "month", "day", "hour", "minute", "second", "weekday", "dayofyear"]

    calculate_features_step = CalculateFeaturesStep(
        datetime_columns=datetime_columns,
        features=features,
    )
    result = calculate_features_step.execute(data)

    assert isinstance(result, DataContainer)
    assert result.X_train["creation_date_year"].equals(
        pd.Series([2023, 2023, 2023, 2023, 2023, 2023, 2024, 2024], dtype="uint16")
    )
    assert result.X_train["creation_date_month"].equals(
        pd.Series([1, 1, 1, 1, 1, 11, 2, 3], dtype="uint8")
    )
    assert result.X_train["creation_date_day"].equals(
        pd.Series([1, 2, 3, 4, 5, 1, 28, 28], dtype="uint8")
    )
    assert result.X_train["creation_date_hour"].equals(
        pd.Series([0, 0, 0, 0, 0, 0, 0, 0], dtype="uint8")
    )
    assert result.X_train["creation_date_minute"].equals(
        pd.Series([0, 0, 0, 0, 0, 0, 0, 0], dtype="uint8")
    )
    assert result.X_train["creation_date_second"].equals(
        pd.Series([0, 0, 0, 0, 0, 0, 0, 0], dtype="uint8")
    )
    assert result.X_train["creation_date_weekday"].equals(
        pd.Series([6, 0, 1, 2, 3, 2, 2, 3], dtype="uint8")
    )
    assert result.X_train["creation_date_dayofyear"].equals(
        pd.Series([1, 2, 3, 4, 5, 305, 59, 88], dtype="uint16")
    )


def test_error_in_incorrect_date_column(data: DataContainer):
    """Check step raises an error when a column has incorrect date values."""
    datetime_columns = ["creation_date", "incorrect_date"]
    features = ["year", "month", "day"]

    calculate_features_step = CalculateFeaturesStep(
        datetime_columns=datetime_columns,
        features=features,
    )

    with pytest.raises(ValueError):
        calculate_features_step.execute(data)


def test_init_with_string_datetime_columns():
    calculate_features_step = CalculateFeaturesStep(
        datetime_columns="creation_date", features=["year"]
    )
    assert calculate_features_step.datetime_columns == ["creation_date"]


def test_init_with_datetime_columns_but_no_features():
    with pytest.raises(ValueError):
        CalculateFeaturesStep(datetime_columns=["creation_date"])


def test_init_with_unsupported_features():
    with pytest.raises(UnsupportedFeatureError):
        CalculateFeaturesStep(datetime_columns=["creation_date"], features=["unsupported_feature"])


def test_execute_with_prediction(data: DataContainer):
    data.is_train = False
    data.X_prediction = data.X_train.copy()

    datetime_columns = ["creation_date"]
    features = ["year", "month", "day"]

    calculate_features_step = CalculateFeaturesStep(
        datetime_columns=datetime_columns,
        features=features,
    )
    result = calculate_features_step.execute(data)

    assert isinstance(result, DataContainer)
    assert "creation_date_year" in result.X_prediction.columns
    assert "creation_date_month" in result.X_prediction.columns
    assert "creation_date_day" in result.X_prediction.columns
