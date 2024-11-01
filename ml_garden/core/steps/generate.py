from enum import Enum
from pathlib import Path
from pprint import pformat
from typing import Optional

import pandas as pd

from ml_garden.core import DataContainer
from ml_garden.core.steps.base import PipelineStep
from ml_garden.utils.df_type_conversions import apply_all_dtype_conversions

# ruff: noqa: FBT001 FBT002 N806 SLF001 C901 PLR0912 PLR0915


class FileType(Enum):
    CSV = ".csv"
    PARQUET = ".parquet"


class GenerateStep(PipelineStep):
    """Generate data from a file."""

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
        **kwargs: Optional[dict],
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
        """Execute the step.

        Parameters
        ----------
        data : DataContainer
            The data container

        Returns
        -------
        DataContainer
            The updated data container
        """
        # Skip GenerateStep if the data is already loaded
        if not data.is_train and data.raw is not None:
            data.flow = data.raw
            return data

        if data.is_train:
            if not self.train_path:
                error_msg = "train_path must be provided for training."
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            if self.test_path:
                self.logger.info("Test path provided: %s", self.test_path)
                # Load test data and add it to the DataContainer
                test_df = self._load_data_from_file(self.test_path)
                data.test = test_df

        if not data.is_train and not self.predict_path and data.raw is None:
            error_msg = (
                "predict_path was not set in the configuration file, and no DataFrame was provided"
                " for prediction. Please provide a predict_path or a DataFrame for prediction."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        file_path = self.train_path if data.is_train else self.predict_path

        self.logger.info("Generating data from file: %s", file_path)

        if not file_path or not Path(file_path).exists():
            error_msg = f"File not found: {file_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        df = self._load_data_from_file(file_path)

        if data.is_train:
            # We'll infer the schema from the concatenation of the train, test and prediction sets
            # (if any).
            # This is to maximize the chance of observing all possible values in each column.
            # For example if an integer column has no NA values in the train set but has a NA on
            # the test set, we need to include it in the schema inference so that the column is
            # assigned a float dtype, instead of int, so that NA values are properly handled
            # Handle the target column separately, since the prediction df won't have a target
            if data.target is None:
                error_msg = "Target not found in DataContainer at Generate step."
                self.logger.error(error_msg)
                raise ValueError(error_msg)

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
                opt_X = opt_X.drop(columns=self.drop_columns)

            if self.optimize_dtypes:
                apply_all_dtype_conversions(df=opt_X, skip_cols=set(self.optimize_dtypes_skip_cols))
                apply_all_dtype_conversions(df=opt_y, skip_cols=set(self.optimize_dtypes_skip_cols))

            # Save the schema for future use in predictions
            data._generate_step_dtypes = opt_X.dtypes.to_dict()
            data._generate_step_dtypes.update(opt_y.dtypes.to_dict())
            if self.train_path.endswith(".csv") or self.optimize_dtypes:
                # Log the inferred schema for csvs or if we optimized dtypes
                self.logger.info(
                    "Inferred Schema for raw data:\n %s", pformat(data._generate_step_dtypes)
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
            if data._generate_step_dtypes is None:
                error_msg = (
                    "Training schema not found in DataContainer. "
                    "Please train the pipeline before making predictions."
                )
                self.logger.error(error_msg)
                raise AttributeError(error_msg)
            for key, value in data._generate_step_dtypes.items():
                try:
                    if key == data.target:
                        # Skip the target column since it's not in the prediction dataframe
                        continue
                    elif key in df.columns:
                        df[key] = df[key].astype(value)
                    else:
                        self._handle_missing_column_error(key)
                except (ValueError, TypeError) as e:
                    self._handle_column_conversion_error(key, value, e)

        df = df.reset_index(drop=True)  # Reset index for consistency
        data.raw = df
        data.flow = df

        self.logger.info("Generated DataFrame with shape: %s", df.shape)

        return data

    def _infer_file_type(self, file_path: str) -> FileType:
        """Infer the file type based on the file extension.

        Parameters
        ----------
        file_path : str
            The file path

        Returns
        -------
        FileType
            The file type
        """
        file_path_obj = Path(file_path)
        file_extension = file_path_obj.suffix.lower()

        try:
            return FileType(file_extension)
        except ValueError:
            error_msg = f"Unsupported file extension: {file_extension}"
            self.logger.exception(error_msg)
            raise ValueError(error_msg) from None

    def _read_csv(self, file_path: str, **kwargs: Optional[dict]) -> pd.DataFrame:
        """Read a CSV file.

        Parameters
        ----------
        file_path : str
            The file path
        **kwargs
            Additional keyword arguments to pass to pd.read_csv

        Returns
        -------
        pd.DataFrame
            The DataFrame
        """
        index_col = kwargs.pop("index", None)
        self.logger.info("Reading CSV file with kwargs: %s", kwargs)
        df = pd.read_csv(file_path, **kwargs)
        if index_col is not None:
            df = df.set_index(index_col)
        return df

    def _read_parquet(self, file_path: str, **kwargs: Optional[dict]) -> pd.DataFrame:
        """Read a parquet file.

        Parameters
        ----------
        file_path : str
            The file path
        **kwargs
            Additional keyword arguments to pass to pd.read_parquet

        Returns
        -------
        pd.DataFrame
            The DataFrame
        """
        index_col = kwargs.pop("index", None)
        self.logger.info("Reading parquet file with kwargs: %s", kwargs)
        df = pd.read_parquet(file_path, **kwargs)
        if index_col is not None:
            df = df.set_index(index_col)  # Avoid using inplace=True
        return df

    def _load_data_from_file(self, file_path: str) -> pd.DataFrame:
        """Load data from a file.

        Parameters
        ----------
        file_path : str
            The file path

        Returns
        -------
        pd.DataFrame
            The DataFrame
        """
        file_type = self._infer_file_type(file_path)

        if file_type == FileType.CSV:
            return self._read_csv(file_path, **self.kwargs)
        elif file_type == FileType.PARQUET:
            return self._read_parquet(file_path, **self.kwargs)
        else:
            error_msg = f"Unsupported file type: {file_type}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _handle_column_conversion_error(self, key: str, value: any, error: Exception) -> None:
        error_msg = (
            f"Failed to convert column {key} to type {value}: {error}. "
            "Verifiy all possible values for the column are observed in the "
            "concatenation of train, test and prediction sets, or specify the dataset "
            "dtypes manually"
        )
        self.logger.error(error_msg)
        raise ValueError(error_msg)

    def _handle_missing_column_error(self, key: str) -> None:
        error_msg = f"Column {key} from training schema not found in DataFrame"
        self.logger.error(error_msg)
        raise ValueError(error_msg)
