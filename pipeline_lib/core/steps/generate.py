import os
from enum import Enum
from typing import Optional

import pandas as pd

from pipeline_lib.core import DataContainer
from pipeline_lib.core.df_type_conversions import apply_all_dtype_conversions
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
        test_path: Optional[str] = None,
        drop_columns: Optional[list[str]] = None,
        optimize_dtypes: bool = False,
        optimize_dtypes_skip_cols: list[str] = [],
        **kwargs,
    ) -> None:
        self.init_logger()
        self.target = target
        self.train_path = train_path
        self.predict_path = predict_path
        self.test_path = test_path
        self.kwargs = kwargs
        self.drop_columns = drop_columns
        self.optimize_dtypes = optimize_dtypes
        self.optimize_dtypes_skip_cols = optimize_dtypes_skip_cols

    def execute(self, data: DataContainer) -> DataContainer:
        """Generate the data from the file."""

        if data.is_train:
            if not self.train_path:
                raise ValueError("train_path must be provided for training.")
            if self.test_path:
                self.logger.info(f"Test path provided: {self.test_path}")
                # Load test data and add it to the DataContainer
                test_df = self._load_data_from_file(self.test_path)
                data.test = test_df

        if not data.is_train and not self.predict_path:
            raise ValueError("predict_path must be provided for prediction.")

        file_path = self.train_path if data.is_train else self.predict_path

        self.logger.info(f"Generating data from file: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        df = self._load_data_from_file(file_path)

        if data.generate_schema is None:
            if self.drop_columns is not None:
                df.drop(columns=self.drop_columns, inplace=True)

            if self.optimize_dtypes:
                apply_all_dtype_conversions(df=df, skip_cols=set(self.optimize_dtypes_skip_cols))

            # Save the schema for future use in predictions
            data.generate_schema = df.dtypes.to_dict()
        else:
            # Apply the saved schema to the DataFrame
            for key, value in data.generate_schema.items():
                df[key] = df[key].astype(value)

        data.raw = df
        data.flow = df

        data.target = self.target

        # remove target if it doesn't exist in the DataFrame for prediction
        if not data.is_train:
            if self.target not in df.columns:
                data.target = None

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

    def _load_data_from_file(self, file_path: str) -> pd.DataFrame:
        file_type = self._infer_file_type(file_path)

        if file_type == FileType.CSV:
            return self._read_csv(file_path, **self.kwargs)
        elif file_type == FileType.PARQUET:
            return self._read_parquet(file_path, **self.kwargs)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
