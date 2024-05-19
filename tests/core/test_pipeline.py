import os

import pytest

from pipeline_lib import Pipeline
from pipeline_lib.implementation.tabular.xgboost.model import XGBoost


@pytest.fixture(autouse=True)
def setup_function():
    Pipeline.model_registry.register_model(XGBoost)


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
