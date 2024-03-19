import os
from enum import Enum

import pandas as pd

from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps.base import PipelineStep


class FileType(Enum):
    CSV = ".csv"
    PARQUET = ".parquet"


class GenerateStep(PipelineStep):
    def __init__(self, path: str, **kwargs):
        self.init_logger()
        self.file_path = path
        self.kwargs = kwargs

    def execute(self, data: DataContainer) -> DataContainer:
        self.logger.info(f"Generating data from file: {self.file_path}")

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        file_type = self._infer_file_type()

        if file_type == FileType.CSV:
            df = self._read_csv()
        elif file_type == FileType.PARQUET:
            df = self._read_parquet()
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        data.raw = df

        self.logger.info(f"Generated DataFrame with shape: {df.shape}")

        return data

    def _infer_file_type(self) -> FileType:
        _, file_extension = os.path.splitext(self.file_path)
        file_extension = file_extension.lower()

        try:
            return FileType(file_extension)
        except ValueError:
            raise ValueError(f"Unsupported file extension: {file_extension}")

    def _read_csv(self) -> pd.DataFrame:
        kwargs = self.kwargs.copy()
        index_col = kwargs.pop("index", None)
        df = pd.read_csv(self.file_path, **kwargs)
        if index_col is not None:
            df.set_index(index_col, inplace=True)
        return df

    def _read_parquet(self) -> pd.DataFrame:
        kwargs = self.kwargs.copy()
        index_col = kwargs.pop("index", None)
        df = pd.read_parquet(self.file_path, **kwargs)
        if index_col is not None:
            df.set_index(index_col, inplace=True)
        return df
