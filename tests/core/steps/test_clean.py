import pandas as pd
import pytest

from ml_garden.core import DataContainer
from ml_garden.core.steps.clean import CleanStep

# Add these constants near the top of the file, after imports


@pytest.fixture
def input_data() -> pd.DataFrame:
    data_dict = {
        "id": [1, 2, 3, 4, 5],
        "column1": [10, 20, None, 40, 50],
        "column2": ["A", "B", "C", "D", "E"],
        "column3": [-100, 2.5, 3.5, 4.5, 250.3],
    }
    return pd.DataFrame(data_dict)


@pytest.fixture
def data(input_data: pd.DataFrame) -> DataContainer:
    data_container = DataContainer({"is_train": True})
    data_container.columns_to_ignore_for_training = []
    data_container.train = input_data
    return data_container


def test_fill_missing(data: DataContainer) -> None:
    clean_step = CleanStep(fill_missing={"column1": 0})
    result = clean_step.execute(data)
    assert result.train["column1"].isna().sum() == 0
    assert result.train["column1"][2] == 0


def test_remove_outliers_clip(data: DataContainer) -> None:
    column = "column3"
    outlier_rows = [-100, 250.3]
    normal_rows = [2.5, 3.5, 4.5]

    clean_step = CleanStep(remove_outliers={column: "clip"})
    result = clean_step.execute(data)

    # Check that the outlier values are dropped from the DataFrame
    for outlier in outlier_rows:
        assert (
            outlier not in result.train[column].to_numpy()
        ), f"Outlier value {outlier} was not dropped from the DataFrame"

    # Check that the normal values are still present in the DataFrame
    for normal_value in normal_rows:
        assert (
            normal_value in result.train[column].to_numpy()
        ), f"Normal value {normal_value} was incorrectly dropped from the DataFrame"


def test_remove_outliers_drop(data: DataContainer) -> None:
    column = "column3"
    outlier_rows = [-100, 250.3]
    normal_rows = [2.5, 3.5, 4.5]

    clean_step = CleanStep(remove_outliers={column: "drop"})
    result = clean_step.execute(data)

    # Check that the outlier values are dropped from the DataFrame
    for outlier in outlier_rows:
        assert (
            outlier not in result.train[column].to_numpy()
        ), f"Outlier value {outlier} was not dropped from the DataFrame"

    # Check that the normal values are still present in the DataFrame
    for normal_value in normal_rows:
        assert (
            normal_value in result.train[column].to_numpy()
        ), f"Normal value {normal_value} was incorrectly dropped from the DataFrame"


def test_convert_dtypes(data: DataContainer) -> None:
    clean_step = CleanStep(convert_dtypes={"column1": "float"})
    result = clean_step.execute(data)
    assert result.train["column1"].dtype == "float"


def test_drop_na_columns(data: DataContainer) -> None:
    expected_length_after_drop_na_columns = 4
    clean_step = CleanStep(drop_na_columns=["column1"])
    result = clean_step.execute(data)
    assert len(result.train) == expected_length_after_drop_na_columns


def test_drop_ids(data: DataContainer) -> None:
    ids_to_drop = [2, 4]
    expected_length_after_drop = 3

    clean_step = CleanStep(drop_ids={"id": ids_to_drop})
    result = clean_step.execute(data)

    assert len(result.train) == expected_length_after_drop
    assert ids_to_drop[0] not in result.train["id"].to_numpy()
    assert ids_to_drop[1] not in result.train["id"].to_numpy()


def test_filter(data: DataContainer) -> None:
    expected_length_after_filter = 2
    column1_threshold = 30

    clean_step = CleanStep(
        filter_conditions={"column1_greater_than_30": f"column1 > {column1_threshold}"}
    )
    result = clean_step.execute(data)

    assert len(result.train) == expected_length_after_filter
    assert result.train["column1"].min() > column1_threshold
