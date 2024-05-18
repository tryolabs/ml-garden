import numpy as np
import pytest

from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps.generate import GenerateStep


# Fixture to create a sample CSV file for testing
@pytest.fixture
def train_csv_file() -> str:
    return "tests/data/train.csv"


@pytest.fixture
def predict_csv_file() -> str:
    return "tests/data/predict.csv"


# Fixture to create a DataContainer for testing
@pytest.fixture
def train_data_container() -> DataContainer:
    data = DataContainer({"target": "target", "is_train": True})
    return data


@pytest.fixture
def predict_data_container() -> DataContainer:
    data = DataContainer({"target": "target", "is_train": False})

    generate_step_dtypes = {
        "date": np.dtype("O"),
        "category_low": np.dtype("O"),
        "category_high": np.dtype("O"),
        "numeric": np.dtype("int64"),
        "target": np.dtype("int64"),
    }
    data._generate_step_dtypes = generate_step_dtypes

    return data


def test_simple_train_csv_execute(train_csv_file: str, train_data_container: DataContainer):
    generate_step = GenerateStep(train_path=train_csv_file)
    result = generate_step.execute(train_data_container)

    assert isinstance(result, DataContainer)
    assert result.raw is not None
    assert result.raw.shape == (8, 5)
    assert list(result.raw.columns) == [
        "date",
        "category_low",
        "category_high",
        "numeric",
        "target",
    ]


def test_simple_predict_csv_execute(predict_csv_file: str, predict_data_container: DataContainer):
    generate_step = GenerateStep(predict_path=predict_csv_file)
    result = generate_step.execute(predict_data_container)

    assert isinstance(result, DataContainer)
    assert result.raw is not None
    assert result.raw.shape == (3, 4)
    assert list(result.raw.columns) == [
        "date",
        "category_low",
        "category_high",
        "numeric",
    ]


def test_mandatory_generate_step_dtypes_in_prediction(predict_csv_file: str):
    generate_step = GenerateStep(predict_path=predict_csv_file)
    data_container = DataContainer({"target": "target", "is_train": False})
    # missing generate_step_dtypes in data_container

    with pytest.raises(AttributeError):
        generate_step.execute(data_container)
