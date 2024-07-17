"""DataContainer class for storing data used in pipeline processing."""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Optional, Union

import dill as pickle
import pandas as pd
from explainerdashboard.explainers import BaseExplainer
from sklearn.compose import ColumnTransformer

from ml_garden.core.constants import Task
from ml_garden.core.model import Model
from ml_garden.utils.compression_utils import compress_zipfile, decompress_zipfile


class DataContainer:
    """
    A container for storing and manipulating data in a pipeline.

    Attributes
    ----------
    data : dict
        A dictionary to store data items.
    """

    def __init__(self, initial_data: Optional[dict] = None):
        """
        Initialize the DataContainer with an empty dictionary or provided data.

        Parameters
        ----------
        initial_data : dict, optional
            Initial data to populate the container.
        """
        self.data = initial_data if initial_data is not None else {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(f"{self.__class__.__name__} initialized")

    def update(self, other: DataContainer) -> None:
        """
        Update the data in this container with another DataContainer's data.

        Parameters
        ----------
        other : DataContainer
            The DataContainer to copy data from.
        """
        self.data.update(other.data)

    def add(self, key: str, value):
        """
        Add a new item to the container.

        Parameters
        ----------
        key : str
            The key under which the value is stored.
        value
            The data to be stored.

        Returns
        -------
        None
        """
        self.data[key] = value
        self.logger.debug(f"Data added under key: {key}")

    def get(self, key: str, default=None):
        """
        Retrieve an item from the container by its key.

        Parameters
        ----------
        key : str
            The key of the item to retrieve.
        default
            The default value to return if the key is not found. Defaults to None.

        Returns
        -------
        The data stored under the given key or the default value.
        """
        return self.data.get(key, default)

    def __getitem__(self, key: str):
        """
        Retrieve an item using bracket notation.

        Parameters
        ----------
        key : str
            The key of the item to retrieve.

        Returns
        -------
        The data stored under the given key.
        """
        return self.get(key)

    def __setitem__(self, key: str, value):
        """
        Add or update an item using bracket notation.

        Parameters
        ----------
        key : str
            The key under which the value is stored.
        value
            The data to be stored.

        Returns
        -------
        None
        """
        self.add(key, value)

    def contains(self, key: str) -> bool:
        """
        Check if the container contains an item with the specified key.

        Parameters
        ----------
        key : str
            The key to check in the container.

        Returns
        -------
        bool
            True if the key exists, False otherwise.
        """
        return key in self.data

    def __contains__(self, key: str) -> bool:
        """
        Enable usage of the 'in' keyword.

        Parameters
        ----------
        key : str
            The key to check in the container.

        Returns
        -------
        bool
            True if the key exists, False otherwise.
        """
        return self.contains(key)

    @property
    def keys(self) -> list[str]:
        """
        Return the keys of the container.

        Returns
        -------
        list[str]
            The keys of the container.
        """
        return list(self.data.keys())

    def save(self, file_path: str, keys: Optional[Union[str, list[str]]] = None):
        """
        Serialize the container data using pickle and save it to a file.

        Parameters
        ----------
        file_path : str
            The path of the file where the serialized data should be saved.
        keys : Optional[Union[str, List[str]]], optional
            The keys of the data to be saved. If None, all data is saved.

        Returns
        -------
        None
        """
        if isinstance(keys, str):
            keys = [keys]

        data_to_save = {k: self.data.get(k) for k in keys} if keys else self.data

        serialized_data = pickle.dumps(data_to_save)

        with open(file_path, "wb") as file:
            file.write(serialized_data)

        compress_zipfile(filename=file_path, delete_uncompressed=True)

        data_size_bytes = sys.getsizeof(serialized_data)
        data_size_mb = data_size_bytes / 1048576  # Convert bytes to megabytes
        disk_size_mb = os.path.getsize(file_path + ".zip") / 1048576  # Convert bytes to megabytes

        self.logger.info(
            f"{self.__class__.__name__} serialized and saved to {file_path}.zip. Serialized Data"
            f" size: {data_size_mb:.2f} MB. Size on Disk: {disk_size_mb:.2f} MB."
        )

    @classmethod
    def from_pickle(
        cls, file_path: str, keys: Optional[Union[str, list[str]]] = None
    ) -> DataContainer:
        """
        Load data from a  pickle file and return a new instance of DataContainer.

        Parameters
        ----------
        file_path : str
            The path of the file from which the serialized data should be read.
        keys : Optional[Union[str, List[str]]], optional
            The keys of the data to be loaded. If None, all data is loaded.

        Returns
        -------
        DataContainer
            A new instance of DataContainer populated with the deserialized data.
        """
        # Check file is a pickle file
        if not file_path.endswith(".pkl"):
            raise ValueError(f"File {file_path} is not a pickle file")

        decompress_zipfile(filename=file_path)

        with open(file_path, "rb") as file:
            data = pickle.loads(file.read())

        os.unlink(file_path)  # Delete the unzipped file after reading

        if isinstance(keys, str):
            keys = [keys]

        if keys:
            data = {k: v for k, v in data.items() if k in keys}

        new_container = cls(initial_data=data)

        if keys:
            loaded_keys = set(new_container.keys)
            not_loaded_keys = set(keys) - loaded_keys if keys else set()
            if not_loaded_keys:
                new_container.logger.warning(f"Keys without values: {not_loaded_keys}")

        new_container.logger.info(f"{cls.__name__} loaded from {file_path}")
        return new_container

    @property
    def clean(self) -> pd.DataFrame:
        """
        Get the clean data from the DataContainer.

        Returns
        -------
        pd.DataFrame
            The clean data stored in the DataContainer.
        """
        return self["clean"]

    @clean.setter
    def clean(self, value: pd.DataFrame):
        """
        Set the clean data in the DataContainer.

        Parameters
        ----------
        value
            The clean data to be stored in the DataContainer.
        """
        self["clean"] = value

    # create the same for raw
    @property
    def raw(self) -> pd.DataFrame:
        """
        Get the raw data from the DataContainer.

        Returns
        -------
        pd.DataFrame
            The raw data stored in the DataContainer.
        """
        return self["raw"]

    @raw.setter
    def raw(self, value: pd.DataFrame):
        """
        Set the raw data in the DataContainer.

        Parameters
        ----------
        value
            The raw data to be stored in the DataContainer.
        """
        self["raw"] = value

    # FIXME: make type hints more specific than Any, however I don't know the return type of the
    # values in dtypes.
    @property
    def _generate_step_dtypes(self) -> dict[str, Any]:
        """
        Get the schema of the raw data produced by the GenerateStep during training.
        This schema is a dictionary mapping dataframe columns (after drops) to their observed types
        during training

        Returns
        -------
        dict[str, Any]
            A dictionary with columns in keys and their training types in values
        """
        return self["_generate_step_dtypes"]

    @_generate_step_dtypes.setter
    def _generate_step_dtypes(self, value: dict[str, Any]):
        """
        Set the schema of the raw data produced by the GenerateStep during training.
        This schema is a dictionary mapping dataframe columns (after drops) to their observed types
        during training

        Returns
        -------
        dict[str, Any]
            A dictionary with columns in keys and their training types in values
        """
        self["_generate_step_dtypes"] = value

    @property
    def split_indices(self) -> dict[str, pd.Index]:
        """
        Get the indices for each split.
        Indices refer to the dataframe used as input for the SplitStep. Users of the library must
        make sure that the indices are valid.

        Returns
        -------
        dict[str, pd.Index]
            A dictionary with keys "train", "validation" and "test", where each key maps to the
            indices of values corresponding to the train, validation and test splits respectively.
            Test set values may be empty
        """
        return self["split_indices"]

    @split_indices.setter
    def split_indices(self, value: dict[str, pd.Index]):
        """
        Set the indices for each split.
        Indices refer to the dataframe used as input for the SplitStep Users of the library must
        make sure that the indices are valid.

        Parameters
        ----------
        value
            A dictionary with keys "train", "validation" and "test", where each key maps to the
            indices of values corresponding to the train, validation and test splits respectively.
            Test set values may be empty
        """
        self["split_indices"] = value

    @property
    def train(self) -> pd.DataFrame:
        """
        Get the train data from the DataContainer.

        Returns
        -------
        pd.DataFrame
            The train data stored in the DataContainer.
        """
        return self["train"]

    @train.setter
    def train(self, value: pd.DataFrame):
        """
        Set the train data in the DataContainer.

        Parameters
        ----------
        value
            The train data to be stored in the DataContainer.
        """
        self["train"] = value

    @property
    def validation(self) -> pd.DataFrame:
        """
        Get the validation data from the DataContainer.

        Returns
        -------
        pd.DataFrame
            The validation data stored in the DataContainer.
        """
        return self["validation"]

    @validation.setter
    def validation(self, value: pd.DataFrame):
        """
        Set the validation data in the DataContainer.

        Parameters
        ----------
        value
            The validation data to be stored in the DataContainer.
        """
        self["validation"] = value

    @property
    def test(self) -> Optional[pd.DataFrame]:
        """
        Get the test data from the DataContainer.

        Returns
        -------
        pd.DataFrame
        The test data stored in the DataContainer.
        """
        return self["test"]

    @test.setter
    def test(self, value: Optional[pd.DataFrame]):
        """
        Set the test data in the DataContainer.

        Parameters
        ----------
        value
        The test data to be stored in the DataContainer.
        """
        self["test"] = value

    @property
    def X_train(self) -> pd.DataFrame:
        """
        Get the encoded training data from the DataContainer. This is data after passing through
        the encoder step, ready to be used as input for the model.

        Returns
        -------
        pd.DataFrame
            The encoded train data stored in the DataContainer.
        """
        return self["X_train"]

    @X_train.setter
    def X_train(self, value: pd.DataFrame):
        """
        Set the encoded training data from the DataContainer. This is data after passing through
        the encoder step, ready to be used as input for the model.

        Parameters
        ----------
        value
            The encoded train data stored in the DataContainer.
        """
        self["X_train"] = value

    @property
    def y_train(self) -> pd.Series:
        """
        Get the encoded training labels from the DataContainer.

        Returns
        -------
        pd.Series
            The encoded training labels stored in the DataContainer.
        """
        return self["y_train"]

    @y_train.setter
    def y_train(self, value: pd.Series):
        """
        Set the encoded training labels to the DataContainer.

        Parameters
        ----------
        value
            The encoded training labels stored in the DataContainer.
        """
        self["y_train"] = value

    @property
    def X_validation(self) -> pd.DataFrame:
        """
        Get the encoded validation data from the DataContainer. This is data after passing through
        the encoder step, ready to be used as input for the model.

        Returns
        -------
        pd.DataFrame
            The encoded validation data stored in the DataContainer.
        """
        return self["X_validation"]

    @X_validation.setter
    def X_validation(self, value: pd.DataFrame):
        """
        Set the encoded validation data from the DataContainer. This is data after passing through
        the encoder step, ready to be used as input for the model.
        Parameters
        ----------
        value
            The encoded validation data stored in the DataContainer.
        """
        self["X_validation"] = value

    @property
    def y_validation(self) -> pd.Series:
        """
        Get the encoded validation labels from the DataContainer.

        Returns
        -------
        pd.Series
            The encoded validation labels stored in the DataContainer.
        """
        return self["y_validation"]

    @y_validation.setter
    def y_validation(self, value: pd.Series):
        """
        Set the encoded validation labels to the DataContainer.
        Parameters
        ----------
        value
            The encoded validation labels stored in the DataContainer.
        """
        self["y_validation"] = value

    @property
    def X_test(self) -> pd.DataFrame:
        """
        Get the encoded test data from the DataContainer. This is data after passing through
        the encoder step, ready to be used as input for the model.

        Returns
        -------
        pd.DataFrame
            The encoded test data stored in the DataContainer.
        """
        return self["X_test"]

    @X_test.setter
    def X_test(self, value: pd.DataFrame):
        """
        Set the encoded test data to the DataContainer. This is data after passing through
        the encoder step, ready to be used as input for the model.

        Parameters
        ----------
        value
            The encoded test data stored in the DataContainer.
        """
        self["X_test"] = value

    @property
    def y_test(self) -> pd.Series:
        """
        Get the encoded test labels from the DataContainer.

        Returns
        -------
        pd.Series
            The encoded test labels stored in the DataContainer.
        """
        return self["y_test"]

    @y_test.setter
    def y_test(self, value: pd.Series):
        """
        Set the encoded test labels to the DataContainer.

        Parameters
        -------
        value
           The encoded test labels stored in the DataContainer.
        """
        self["y_test"] = value

    @property
    def X_prediction(self) -> pd.DataFrame:
        """
        Get the encoded prediction data from the DataContainer. This is data after passing through
        the encoder step, ready to be used as input for the model.

        Returns
        -------
        pd.DataFrame
            The encoded prediction data stored in the DataContainer.
        """
        return self["X_prediction"]

    @X_prediction.setter
    def X_prediction(self, value: pd.DataFrame):
        """
        Set the encoded prediction data to the DataContainer. This is data after passing through
        the encoder step, ready to be used as input for the model.

        Parameters
        -------
        value
           The encoded prediction data stored in the DataContainer.
        """
        self["X_prediction"] = value

    @property
    def model(self) -> Model:
        """
        Get the model from the DataContainer.

        Returns
        -------
        Model
            The model stored in the DataContainer.
        """
        return self["model"]

    @model.setter
    def model(self, value: Model):
        """
        Set the model in the DataContainer.

        Parameters
        ----------
        value
            The model to be stored in the DataContainer.
        """
        self["model"] = value

    @property
    def metrics(self) -> dict:
        """
        Get the metrics from the DataContainer.

        Returns
        -------
        dict
            The metrics stored in the DataContainer.
        """
        return self["metrics"]

    @metrics.setter
    def metrics(self, value: dict):
        """
        Set the metrics in the DataContainer.

        Parameters
        ----------
        dict
            The metrics to be stored in the DataContainer.
        """
        self["metrics"] = value

    @property
    def predictions(self) -> pd.Series:
        """
        Get the predictions from the DataContainer.

        Returns
        -------
        pd.Series
            The predictions stored in the DataContainer.
        """
        return self["predictions"]

    @predictions.setter
    def predictions(self, value: pd.Series):
        """
        Set the predictions in the DataContainer.

        Parameters
        ----------
        pd.Series
            The predictions to be stored in the DataContainer.
        """
        self["predictions"] = value

    @property
    def predict_proba(self) -> pd.DataFrame:
        """
        Get the prediction probabilities from the DataContainer.

        Returns
        -------
        pd.DataFrame
            The prediction probabilities stored in the DataContainer.
        """
        return self["predict_proba"]

    @predict_proba.setter
    def predict_proba(self, value: pd.DataFrame):
        """
        Set the prediction probabilities in the DataContainer.

        Parameters
        ----------
        value : pd.DataFrame
            The prediction probabilities to be stored in the DataContainer.
            Should be a DataFrame with a column for each class.
        """
        self["predict_proba"] = value

    @property
    def explainer(self) -> BaseExplainer:
        """
        Get the explainer from the DataContainer.

        Returns
        -------
        BaseExplainer
            The explainer stored in the DataContainer.
        """
        if not self.is_train:
            raise ValueError(
                "Explainer is only available for training. Pipeline was executed on prediction"
                " mode."
            )
        return self["explainer"]

    @explainer.setter
    def explainer(self, value: BaseExplainer):
        """
        Set the explainer in the DataContainer.

        Parameters
        ----------
        value
            The explainer to be stored in the DataContainer.
        """
        self["explainer"] = value

    @property
    def tuning_params(self) -> dict:
        """
        Get the tuning parameters from the DataContainer.

        Returns
        -------
        dict
            The tuning parameters stored in the DataContainer.
        """
        return self["tuning_params"]

    @tuning_params.setter
    def tuning_params(self, value: dict):
        """
        Set the tuning parameters in the DataContainer.

        Parameters
        ----------
        dict
            The tuning parameters to be stored in the DataContainer.
        """
        self["tuning_params"] = value

    @property
    def target(self) -> str:
        """
        Get the target from the DataContainer.

        Returns
        -------
        str
            The target stored in the DataContainer.
        """
        return self["target"]

    @target.setter
    def target(self, value: str):
        """
        Set the target in the DataContainer.

        Parameters
        ----------
        value
            The target to be stored in the DataContainer.
        """
        self["target"] = value

    @property
    def prediction_column(self) -> str:
        """
        Get the prediction column name from the DataContainer.
        """
        return self["prediction_column"]

    @prediction_column.setter
    def prediction_column(self, value: str):
        """
        Set the prediction column name in the DataContainer.
        """
        self["prediction_column"] = value

    @property
    def columns_to_ignore_for_training(self) -> list[str]:
        """
        Get the columns to ignore for training from the DataContainer.
        This is useful for specifying id columns, which we don't want to include in the model
        training to avoid overfitting, but we want to keep them in the pipeline for metric/reporting
        calculation.

        Returns
        -------
        str
            The columns to ignore for training stored in the DataContainer.
        """
        return self["columns_to_ignore_for_training"]

    @columns_to_ignore_for_training.setter
    def columns_to_ignore_for_training(self, value: list[str]):
        """
        Set the columns to ignore for training from the DataContainer.
        This is useful for specifying id columns, which we don't want to include in the model
        training to avoid overfitting, but we want to keep them in the pipeline for metric/reporting
        calculation.

        Returns
        -------
        str
            The columns to ignore for training stored in the DataContainer.
        """
        self["columns_to_ignore_for_training"] = value

    @property
    def flow(self) -> pd.DataFrame:
        """
        Get the flow from the DataContainer.

        Returns
        -------
        pd.DataFrame
            The flow stored in the DataContainer.
        """
        return self["flow"]

    @flow.setter
    def flow(self, value: pd.DataFrame):
        """
        Set the flow in the DataContainer.

        Parameters
        ----------
        value
            The flow to be stored in the DataContainer.
        """
        self["flow"] = value

    @property
    def is_train(self) -> bool:
        """
        Check if the DataContainer is made for training.

        Returns
        -------
        bool
            True if the DataContainer contains training data, False otherwise.
        """
        return self["is_train"]

    @is_train.setter
    def is_train(self, value: bool):
        """
        Set the is_train flag in the DataContainer.

        Parameters
        ----------
        value
            The is_train flag to be stored in the DataContainer.
        """
        self["is_train"] = value

    @property
    def _encoder(self) -> ColumnTransformer:
        """
        Get the encoder from the DataContainer.

        Returns
        -------
        ColumnTransformer
            The encoder stored in the DataContainer.
        """
        return self["encoder"]

    @_encoder.setter
    def _encoder(self, value: ColumnTransformer):
        """
        Set the encoder in the DataContainer.

        Parameters
        ----------
        value
            The encoder to be stored in the DataContainer.
        """
        self["encoder"] = value

    @property
    def feature_importance(self) -> pd.DataFrame:
        """
        Get the feature_importance from the DataContainer.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing feature importance values.
        """
        return self["feature_importance"]

    @feature_importance.setter
    def feature_importance(self, value: pd.DataFrame):
        """
        Set the feature_importance in the DataContainer.

        Parameters
        ----------
        value
            The feature importance DataFrame to be stored in the DataContainer.
        """
        self["feature_importance"] = value

    @property
    def task(self) -> Task:
        """
        Get the task from the DataContainer.

        Returns
        -------
        Task
            The task, either "regression" or "classification".
        """
        return self["task"]

    @task.setter
    def task(self, value: Task) -> None:
        """
        Set the task in the DataContainer.

        Parameters
        ----------
        value : Task
            The task to set. Must be one of "regression" or "classification".

        Raises
        ------
        ValueError
            If the value is not "regression" or "classification".
        """
        if not isinstance(value, Task):
            raise ValueError(f"task must be an instance of Task enum, got {type(value)}")

        self["task"] = value

    def __eq__(self, other) -> bool:
        """
        Compare this DataContainer with another for equality.

        Parameters
        ----------
        other : DataContainer
            Another DataContainer instance to compare with.

        Returns
        -------
        bool
            True if containers are equal, False otherwise.
        """
        if isinstance(other, DataContainer):
            return self.data == other.data
        return False

    def __ne__(self, other) -> bool:
        """
        Compare this DataContainer with another for inequality.

        Parameters
        ----------
        other : DataContainer
            Another DataContainer instance to compare with.

        Returns
        -------
        bool
            True if containers are not equal, False otherwise.
        """
        return not self.__eq__(other)

    def __str__(self):
        """
        Generate a user-friendly JSON string representation of the DataContainer.

        Returns
        -------
        str
            A JSON string describing the keys and types of contents of the DataContainer.
        """
        data_summary = {key: type(value).__name__ for key, value in self.data.items()}
        return json.dumps(data_summary, indent=4)

    def __repr__(self):
        """
        Generate an official string representation of the DataContainer.

        Returns
        -------
        str
            A formal string representation of the DataContainer's state.
        """
        return f"<DataContainer({self.data})>"
