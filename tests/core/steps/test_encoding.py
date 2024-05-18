import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps import EncodeStep


def train_data() -> pd.DataFrame:
    # Data as a dictionary
    data = {
        "year": np.array([2023, 2023, 2023, 2023, 2023, 2023, 2024, 2024], dtype=np.int64),
        "month": np.array([1, 1, 1, 1, 1, 11, 2, 3], dtype=np.int64),
        "day": np.array([1, 2, 3, 4, 5, 1, 28, 28], dtype=np.int64),
        "category_low": np.array(["A", "B", "A", "A", "B", "A", "B", "A"], dtype=str),
        "category_high": np.array(["X1", "X2", "X3", "X4", "X5", "X6", "X6", "X7"], dtype=str),
        "numeric": np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int64),
        "target": np.array([0, 1, 1, 0, 1, 0, 1, 0], dtype=np.int64),
    }

    # Create the DataFrame
    return pd.DataFrame(data)


# Fixture to create a DataContainer for testing
@pytest.fixture
def train_data_container() -> DataContainer:
    data = DataContainer({"target": "target", "is_train": True})
    data.columns_to_ignore_for_training = []
    data.train = train_data()
    return data


def test_check_dtypes(train_data_container: DataContainer):
    """Test to check data types after encoding."""
    encode_step = EncodeStep()
    result = encode_step.execute(train_data_container)

    assert isinstance(result, DataContainer)
    assert result.X_train.shape == (8, 6)
    assert result.y_train.shape == (8,)
    assert result.X_train["year"].dtype == np.dtype("int64")
    assert result.X_train["month"].dtype == np.dtype("int64")
    assert result.X_train["day"].dtype == np.dtype("int64")
    assert result.X_train["numeric"].dtype == np.dtype("int64")
    assert result.X_train["category_low"].dtype == np.dtype("uint8")  # optimizing int64 to uint8
    assert result.X_train["category_high"].dtype == np.dtype("float32")  # optimizing to float32


def test_check_numeric_passthrough(train_data_container: DataContainer):
    """Test to check if numeric columns are correctly passed through."""
    encode_step = EncodeStep()
    result = encode_step.execute(train_data_container)

    # Checking numeric passthrough
    expected_years = pd.Series(
        [2023, 2023, 2023, 2023, 2023, 2023, 2024, 2024], dtype=np.int64, name="year"
    )
    expected_months = pd.Series([1, 1, 1, 1, 1, 11, 2, 3], dtype=np.int64, name="month")
    expected_days = pd.Series([1, 2, 3, 4, 5, 1, 28, 28], dtype=np.int64, name="day")
    expected_numeric = pd.Series([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int64, name="numeric")

    pdt.assert_series_equal(result.X_train["year"], expected_years)
    pdt.assert_series_equal(result.X_train["month"], expected_months)
    pdt.assert_series_equal(result.X_train["day"], expected_days)
    pdt.assert_series_equal(result.X_train["numeric"], expected_numeric)


def test_check_ordinal_encoding(train_data_container: DataContainer):
    """Test to check if ordinal encoding is correctly applied."""
    encode_step = EncodeStep()
    result = encode_step.execute(train_data_container)

    # Checking ordinal encoding
    expected_category_low = pd.Series([0, 1, 0, 0, 1, 0, 1, 0], dtype=np.uint8, name="category_low")

    pdt.assert_series_equal(result.X_train["category_low"], expected_category_low)


def test_check_target_encoding(train_data_container: DataContainer):
    """Test to check if target encoding is correctly applied."""
    encode_step = EncodeStep()
    result = encode_step.execute(train_data_container)

    # Expected values for 'category_high'
    expected_category_high = pd.Series(
        [0.434946, 0.565054, 0.565054, 0.434946, 0.565054, 0.500000, 0.500000, 0.434946],
        dtype="float32",
        name="category_high",
    )

    pdt.assert_series_equal(
        result.X_train["category_high"],
        expected_category_high,
        check_exact=False,
        rtol=1e-5,
        atol=1e-8,
    )


def test_check_cardinality_threshold(train_data_container: DataContainer):
    """Test to check if cardinality threshold is correctly applied."""
    encode_step = EncodeStep(cardinality_threshold=7)
    result = encode_step.execute(train_data_container)

    # check that the dtype for 'category_high' is now uint8
    assert result.X_train["category_high"].dtype == np.dtype("uint8")


def test_custom_feature_encoders_dictionary(train_data_container: DataContainer):
    """Test to check if custom feature encoders dictionary is correctly applied."""
    custom_encoding = {
        "category_high": {
            "encoder": "OrdinalEncoder",
            "params": {
                "handle_unknown": "use_encoded_value",
            },
        }
    }
    encode_step = EncodeStep(feature_encoders=custom_encoding)
    result = encode_step.execute(train_data_container)

    # check that the dtype for 'category_high' is now uint8
    assert result.X_train["category_high"].dtype == np.dtype("uint8")
