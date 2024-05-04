import os
from enum import Enum
from pprint import pformat
from typing import Optional

import pandas as pd

from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps.base import PipelineStep
from pipeline_lib.utils.df_type_conversions import apply_all_dtype_conversions


class FileType(Enum):
    CSV = ".csv"
    PARQUET = ".parquet"


class GenerateStep(PipelineStep):
    used_for_prediction = True
    used_for_training = True

    def __init__(
        self,
        train_path: Optional[str] = None,
        test_path: Optional[str] = None,
        predict_path: Optional[str] = None,
        drop_columns: Optional[list[str]] = None,
        optimize_dtypes: bool = False,
        optimize_dtypes_skip_cols: Optional[list[str]] = None,
        **kwargs,
    ) -> None:
        self.init_logger()
        self.train_path = train_path
        self.test_path = test_path
        self.predict_path = predict_path
        self.kwargs = kwargs
        self.drop_columns = drop_columns
        self.optimize_dtypes = optimize_dtypes
        self.optimize_dtypes_skip_cols = optimize_dtypes_skip_cols or []

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

        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        df = self._load_data_from_file(file_path)

        if data.is_train:
            # We'll infer the schema from the concatenation of the train, test and prediction sets
            # (if any).
            # This is to maximize the chance of observing all possible values in each column.
            # For example if an integer column has no NA values in the train set but has a NA on
            # the test set, we need to include it in the schema inference so that the column is
            # assigned a float dtype, instead of int, so that NA values are properly handled
            # Handle the target column separately, since the prediction df won't have a target
            opt_X = pd.concat([df, data.test]) if data.test is not None else df
            opt_y = opt_X[[data.target]]
            opt_X = opt_X.drop(columns=[data.target])

            if self.predict_path:
                predict_df = self._load_data_from_file(self.predict_path)
                if data.target in predict_df.columns:
                    opt_y = pd.concat([opt_y, predict_df[[data.target]]])
                    opt_X = pd.concat([opt_X, predict_df.drop(columns=[data.target])])
                else:
                    opt_X = pd.concat([opt_X, predict_df])

            if self.drop_columns is not None:
                opt_X.drop(columns=self.drop_columns, inplace=True)

            if self.optimize_dtypes:
                apply_all_dtype_conversions(df=opt_X, skip_cols=set(self.optimize_dtypes_skip_cols))
                apply_all_dtype_conversions(df=opt_y, skip_cols=set(self.optimize_dtypes_skip_cols))

            # Save the schema for future use in predictions
            data._generate_step_dtypes = opt_X.dtypes.to_dict()
            data._generate_step_dtypes.update(opt_y.dtypes.to_dict())
            if self.train_path.endswith(".csv") or self.optimize_dtypes:
                # Log the inferred schema for csvs or if we optimized dtypes
                self.logger.info(
                    f"Inferred Schema for raw data:\n {pformat(data._generate_step_dtypes)}"
                )

            # Re-split the optimized df into train/test, discard prediction since we're doing
            # training for now
            i_max_row = len(df) + len(data.test) if data.test is not None else len(df)
            opt_X = opt_X.iloc[:i_max_row, :]
            opt_y = opt_y.iloc[:i_max_row, :]
            opt_X = pd.concat([opt_X, opt_y], axis=1)
            df = opt_X.iloc[0 : len(df)]
            if data.test is not None:
                data.test = opt_X.iloc[len(df) :]
        else:
            # Apply the schema saved during training to the DataFrame
            for key, value in data._generate_step_dtypes.items():
                try:
                    if key in df.columns:
                        df[key] = df[key].astype(value)
                    elif key != data.target:
                        # Target column may not be in the prediction dataframe
                        raise ValueError(
                            f"Column {key} from training schema not found in DataFrame"
                        )
                except Exception as e:
                    raise ValueError(
                        f"Failed to convert column {key} to type {value}: {e}. "
                        "Verifiy all possible values for the column are observed in the "
                        "concatenation of train, test and prediction sets, or specify the dataset "
                        "dtypes manually"
                    )

        df.reset_index(drop=True, inplace=True)  # Reset index for consistency
        data.raw = df
        data.flow = df

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
