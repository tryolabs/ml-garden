"""DataContainer class for storing data used in pipeline processing."""

from __future__ import annotations

import json
import logging
import pickle
import sys
from typing import Optional, Union

import yaml


class DataContainer:
    """
    A container for storing and manipulating data in a pipeline.

    Attributes
    ----------
    data : dict
        A dictionary to store data items.
    """

    GENERATE_CONFIGS = "generate_configs"
    CLEAN_CONFIGS = "clean_configs"
    SPLIT_CONFIGS = "split_configs"
    TARGET_SCALING_CONFIGS = "target_scaling_configs"
    RAW = "raw"
    CLEAN = "clean"
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    MODEL = "model"
    MODEL_CONFIGS = "model_configs"
    MODEL_INPUT = "model_input"
    MODEL_OUTPUT = "model_output"
    METRICS = "metrics"
    PREDICTIONS = "predictions"
    EXPLAINER = "explainer"
    TUNING_PARAMS = "tuning_params"

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

        data_to_save = {k: self.data[k] for k in keys} if keys else self.data

        serialized_data = pickle.dumps(data_to_save)
        data_size_bytes = sys.getsizeof(serialized_data)
        data_size_mb = data_size_bytes / 1048576  # Convert bytes to megabytes

        with open(file_path, "wb") as file:
            file.write(serialized_data)
        self.logger.info(
            f"{self.__class__.__name__} serialized and saved to {file_path}. Size:"
            f" {data_size_mb:.2f} MB"
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

        with open(file_path, "rb") as file:
            data = pickle.loads(file.read())

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

    @classmethod
    def from_json(cls, file_path: str) -> DataContainer:
        """
        Create a new DataContainer instance from a JSON file.

        Parameters
        ----------
        file_path : str
            The path to the JSON file containing the configurations.

        Returns
        -------
        DataContainer
            A new instance of DataContainer populated with the data from the JSON file.
        """
        # Check file is a JSON file
        if not file_path.endswith(".json"):
            raise ValueError(f"File {file_path} is not a JSON file")

        with open(file_path, "r") as f:
            data = json.load(f)

        # The loaded data is used as the initial data for the DataContainer instance
        return cls(initial_data=data)

    @classmethod
    def from_yaml(cls, file_path: str) -> DataContainer:
        """
        Create a new DataContainer instance from a YAML file.

        Parameters
        ----------
        file_path : str
            The path to the YAML file containing the configurations.

        Returns
        -------
        DataContainer
            A new instance of DataContainer populated with the data from the YAML file.

        Raises
        ------
        ValueError
            If the provided file is not a YAML file.
        """
        # Check if the file has a .yaml or .yml extension
        if not (file_path.endswith(".yaml") or file_path.endswith(".yml")):
            raise ValueError(f"The file {file_path} is not a YAML file.")

        try:
            with open(file_path, "r") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            # Handle cases where the file content is not valid YAML
            raise ValueError(f"Error parsing YAML from {file_path}: {e}")

        # The loaded data is used as the initial data for the DataContainer instance
        return cls(initial_data=data)

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
