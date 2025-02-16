import numpy as np
import pandas as pd
import pytest

from ml_garden.core import DataContainer
from ml_garden.core.random_state_generator import RandomStateManager
from ml_garden.core.steps import TabularSplitStep

# ruff: noqa: ERA001


@pytest.fixture(autouse=True)
def _setup_random_state() -> None:
    """Initialize random state before each test."""
    RandomStateManager.initialize(seed=42)


@pytest.fixture()
def input_data() -> pd.DataFrame:
    # Data as a dictionary
    data_dict = {
        "year": np.array([2023, 2023, 2023, 2023, 2023, 2023, 2024, 2024], dtype=np.int64),
        "month": np.array([1, 1, 1, 1, 1, 11, 2, 3], dtype=np.int64),
        "day": np.array([1, 2, 3, 4, 5, 1, 28, 28], dtype=np.int64),
        "category_low": np.array(["A", "B", "A", "A", "B", "A", "B", "A"], dtype=str),
        "category_high": np.array(["X1", "X2", "X3", "X4", "X5", "X6", "X6", "X7"], dtype=str),
        "numeric": np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int64),
        "target": np.array([0, 1, 1, 0, 1, 0, 1, 0], dtype=np.int64),
    }

    # Create the DataFrame
    return pd.DataFrame(data_dict)


@pytest.fixture()
def data(input_data: pd.DataFrame) -> DataContainer:
    data_container = DataContainer({"is_train": True})
    data_container.columns_to_ignore_for_training = []
    data_container.flow = input_data
    return data_container


def test_train_val_percentage(data: DataContainer) -> None:
    """Test to check if the train and val percentage is correctly applied."""
    split_step = TabularSplitStep(train_percentage=0.8)
    result = split_step.execute(data)

    assert isinstance(result, DataContainer)
    assert result.train.shape == (6, 7)
    assert result.validation.shape == (2, 7)


def test_train_val_test_percentage(data: DataContainer) -> None:
    """Test to check if the train and val percentage is correctly applied."""
    split_step = TabularSplitStep(
        train_percentage=0.8,
        validation_percentage=0.1,
        test_percentage=0.1,
    )
    result = split_step.execute(data)

    assert isinstance(result, DataContainer)
    assert result.train.shape == (6, 7)
    assert result.validation.shape == (1, 7)
    assert result.test.shape == (1, 7)


def test_that_test_percentage_is_needed_with_validation() -> None:
    """Test to check if the test percentage is needed when validation percentage is provided."""
    with pytest.raises(
        ValueError,
        match="The sum of train_percentage and validation_percentage must equal 1 when "
        "test_percentage is not specified.",
    ):
        split_step = TabularSplitStep(  # noqa: F841
            train_percentage=0.8,
            validation_percentage=0.1,
        )


def test_sum_of_percentages() -> None:
    """Test to check if the sum of percentages is not more than 1."""
    with pytest.raises(
        ValueError,
        match="The sum of train_percentage, validation_percentage, and test_percentage must "
        "equal 1",
    ):
        split_step = TabularSplitStep(  # noqa: F841
            train_percentage=0.8,
            validation_percentage=0.2,
            test_percentage=0.2,
        )


# TODO(@diegomarvid): fix this test, it is failing
# https://github.com/tryolabs/ml-garden/issues/37
# def test_group_by_columns_single_column() -> None:
#     """Test to check if the group by columns is correctly applied."""
#     dataset = input_data()
#     data_container = DataContainer({"is_train": True})
#     data_container.flow = dataset

#     split_step = TabularSplitStep(train_percentage=0.7, group_by_columns=["category_low"])
#     result = split_step.execute(data_container)

#     assert isinstance(result, DataContainer)
#     assert result.train.shape == (5, 7)
#     assert result.validation.shape == (3, 7)
#     assert len(result.train["category_low"].unique()) == 1
#     assert len(result.validation["category_low"].unique()) == 1
