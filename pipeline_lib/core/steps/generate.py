import os
from enum import Enum
from typing import Optional

import pandas as pd

from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps.base import PipelineStep


class FileType(Enum):
    CSV = ".csv"
    PARQUET = ".parquet"


class GenerateStep(PipelineStep):
    used_for_prediction = True
    used_for_training = True

    def __init__(
        self,
        target: str,
        train_path: Optional[str] = None,
        predict_path: Optional[str] = None,
        **kwargs,
    ):
        self.init_logger()
        self.target = target
        self.train_path = train_path
        self.predict_path = predict_path
        self.kwargs = kwargs

    def execute(self, data: DataContainer) -> DataContainer:
        """Generate the data from the file."""

        if data.is_train and not self.train_path:
            raise ValueError("train_path must be provided for training.")

        if not data.is_train and not self.predict_path:
            raise ValueError("predict_path must be provided for prediction.")

        file_path = self.train_path if data.is_train else self.predict_path

        self.logger.info(f"Generating data from file: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_type = self._infer_file_type(file_path)

        kwargs = self.kwargs.copy()
        if data.is_train:
            kwargs.pop("predict_path", None)  # Remove prediction_path if in train mode
        else:
            kwargs.pop("train_path", None)  # Remove train_path if in predict mode

        if file_type == FileType.CSV:
            df = self._read_csv(file_path, **kwargs)
        elif file_type == FileType.PARQUET:
            df = self._read_parquet(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        data.raw = df
        data.flow = df
        data.target = self.target

        self.logger.info(f"Generated DataFrame with shape: {df.shape}")

        return data

    def _infer_file_type(self, file_path: str) -> FileType:
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()

        try:
            return FileType(file_extension)
        except ValueError:
            raise ValueError(f"Unsupported file extension: {file_extension}")

    def _read_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        index_col = kwargs.pop("index", None)
        self.logger.info(f"Reading CSV file with kwargs: {kwargs}")
        df = pd.read_csv(file_path, **kwargs)
        if index_col is not None:
            df.set_index(index_col, inplace=True)
        return df

    def _read_parquet(self, file_path: str, **kwargs) -> pd.DataFrame:
        index_col = kwargs.pop("index", None)
        self.logger.info(f"Reading parquet file with kwargs: {kwargs}")
        df = pd.read_parquet(file_path, **kwargs)
        if index_col is not None:
            df.set_index(index_col, inplace=True)
        return df
