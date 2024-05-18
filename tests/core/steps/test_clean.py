import pandas as pd
import pytest

from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps.clean import CleanStep


@pytest.fixture
def input_data() -> pd.DataFrame:
    data = {
        "id": [1, 2, 3, 4, 5],
        "column1": [10, 20, None, 40, 50],
        "column2": ["A", "B", "C", "D", "E"],
        "column3": [-100, 2.5, 3.5, 4.5, 250.3],
    }
    return pd.DataFrame(data)


@pytest.fixture
def data(input_data: pd.DataFrame) -> DataContainer:
    data = DataContainer({"is_train": True})
    data.columns_to_ignore_for_training = []
    data.train = input_data
    return data


def test_fill_missing(data):
    clean_step = CleanStep(fill_missing={"column1": 0})
    result = clean_step.execute(data)
    assert result.train["column1"].isna().sum() == 0
    assert result.train["column1"][2] == 0


def test_remove_outliers_clip(data):
    column = "column3"
    outlier_rows = [-100, 250.3]
    normal_rows = [2.5, 3.5, 4.5]

    clean_step = CleanStep(remove_outliers={column: "clip"})
    result = clean_step.execute(data)

    # Check that the outlier values are dropped from the DataFrame
    for outlier in outlier_rows:
        assert (
            outlier not in result.train[column].values
        ), f"Outlier value {outlier} was not dropped from the DataFrame"

    # Check that the normal values are still present in the DataFrame
    for normal_value in normal_rows:
        assert (
            normal_value in result.train[column].values
        ), f"Normal value {normal_value} was incorrectly dropped from the DataFrame"


def test_remove_outliers_drop(data):
    column = "column3"
    outlier_rows = [-100, 250.3]
    normal_rows = [2.5, 3.5, 4.5]

    clean_step = CleanStep(remove_outliers={column: "drop"})
    result = clean_step.execute(data)

    # Check that the outlier values are dropped from the DataFrame
    for outlier in outlier_rows:
        assert (
            outlier not in result.train[column].values
        ), f"Outlier value {outlier} was not dropped from the DataFrame"

    # Check that the normal values are still present in the DataFrame
    for normal_value in normal_rows:
        assert (
            normal_value in result.train[column].values
        ), f"Normal value {normal_value} was incorrectly dropped from the DataFrame"


def test_convert_dtypes(data):
    clean_step = CleanStep(convert_dtypes={"column1": "float"})
    result = clean_step.execute(data)
    assert result.train["column1"].dtype == "float"


def test_drop_na_columns(data):
    clean_step = CleanStep(drop_na_columns=["column1"])
    result = clean_step.execute(data)
    assert len(result.train) == 4


def test_drop_ids(data):
    clean_step = CleanStep(drop_ids={"id": [2, 4]})
    result = clean_step.execute(data)
    assert len(result.train) == 3
    assert 2 not in result.train["id"].values
    assert 4 not in result.train["id"].values


def test_filter(data):
    clean_step = CleanStep(filter={"column1_greater_than_30": "column1 > 30"})
    result = clean_step.execute(data)
    assert len(result.train) == 2
    assert result.train["column1"].min() > 30
