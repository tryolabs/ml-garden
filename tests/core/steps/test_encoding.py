import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from ml_garden.core import DataContainer
from ml_garden.core.steps import EncodeStep


@pytest.fixture
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
def train_data_container(train_data: pd.DataFrame) -> DataContainer:
    data = DataContainer({"target": "target", "is_train": True})
    data.columns_to_ignore_for_training = []
    data.X_train = train_data.drop(columns=["target"])
    data.y_train = train_data["target"]
    return data


def test_check_numeric_passthrough(train_data_container: DataContainer):
    """Test to check if numeric columns are correctly passed through."""
    encode_step = EncodeStep()
    result = encode_step.execute(train_data_container)

    numeric_columns = ["year", "month", "day", "numeric"]

    for column in numeric_columns:
        # Check that the numeric column is present in the encoded data
        assert column in result.X_train.columns

        # Check that the dtype of the numeric column remains the same
        assert result.X_train[column].dtype == train_data_container.X_train[column].dtype

        # Check that the values of the numeric column remain unchanged
        pdt.assert_series_equal(result.X_train[column], train_data_container.X_train[column])


def test_check_ordinal_encoding(train_data_container: DataContainer):
    """Test to check if ordinal encoding is correctly applied."""
    encode_step = EncodeStep()
    result = encode_step.execute(train_data_container)

    # Check that the 'category_low' column is encoded as uint8
    assert result.X_train["category_low"].dtype == np.dtype("uint8")

    # Check that the encoded values are integers starting from 0
    assert result.X_train["category_low"].between(0, result.X_train["category_low"].max()).all()

    # Check that the number of unique encoded values matches the number of unique categories
    assert len(result.X_train["category_low"].unique()) == len(
        train_data_container.X_train["category_low"].unique()
    )


def test_check_target_encoding(train_data_container: DataContainer):
    """Test to check if target encoding is correctly applied."""
    encode_step = EncodeStep()
    result = encode_step.execute(train_data_container)

    # Check that the 'category_high' column is encoded as float32
    assert result.X_train["category_high"].dtype == np.dtype("float32")

    # Check that the encoded values are within the expected range [0, 1]
    assert result.X_train["category_high"].between(0, 1).all()

    # Check that the encoded values are different for different categories
    assert len(result.X_train["category_high"].unique()) > 1


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
