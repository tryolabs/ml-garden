"""Calculate datetime-related features from specified columns."""

from typing import Optional, Union

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from ml_garden.core import DataContainer
from ml_garden.core.steps.base import PipelineStep


class UnsupportedFeatureError(Exception):
    """Custom exception for unsupported features."""


class CalculateFeaturesStep(PipelineStep):
    """Calculate datetime-related features from specified columns."""

    used_for_prediction = True
    used_for_training = True

    def __init__(
        self,
        datetime_columns: Optional[Union[list[str], str]] = None,
        features: Optional[list[str]] = None,
    ) -> None:
        """Initialize CalculateFeaturesStep.

        Parameters
        ----------
        datetime_columns : Union[List[str], str], optional
            The name of the column or columns containing datetime values, by default None
        features : Optional[List[str]], optional
        """
        super().__init__()
        self.init_logger()
        self.datetime_columns = datetime_columns
        self.features = features

        if self.datetime_columns and not isinstance(self.datetime_columns, list):
            self.datetime_columns = [self.datetime_columns]

        self.feature_extractors = {
            "year": lambda col: col.dt.year,
            "month": lambda col: col.dt.month,
            "day": lambda col: col.dt.day,
            "hour": lambda col: col.dt.hour,
            "minute": lambda col: col.dt.minute,
            "second": lambda col: col.dt.second,
            "weekday": lambda col: col.dt.weekday,
            "dayofyear": lambda col: col.dt.dayofyear,
        }

        # Validate features during initialization
        if self.features:
            unsupported_features = set(self.features) - set(self.feature_extractors.keys())
            if unsupported_features:
                unsupported_features_message = (
                    f"Unsupported datetime features: {unsupported_features}"
                )
                raise UnsupportedFeatureError(unsupported_features_message)

        if self.datetime_columns and not self.features:
            message = (
                "No datetime features specified. Must specify at least one feature. Possible"
                f" features: {list(self.feature_extractors.keys())}"
            )
            raise ValueError(message)

    def _convert_column_to_datetime(self, df: pd.DataFrame, column: str, log: bool) -> pd.DataFrame:
        """Convert a column to datetime.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the column to convert
        column : str
            The name of the column to convert
        log: bool
            If True, logs information.

        Returns
        -------
        pd.DataFrame
            The DataFrame with the column converted to datetime
        """
        # Check if the column is already a datetime type
        if not is_datetime64_any_dtype(df[column]):
            try:
                df[column] = pd.to_datetime(
                    df[column],
                    errors="raise",
                )
                if log:
                    self.logger.info(f"Column '{column}' automatically converted to datetime.")
            except ValueError:
                self.logger.exception(f"Error converting column '{column}' to datetime")
            except Exception:
                self.logger.exception(f"Unexpected error converting column '{column}' to datetime")
        elif log:
            self.logger.debug(f"Column '{column}' is already a datetime type.")
        return df

    def _extract_feature(self, df: pd.DataFrame, column: str, feature: str) -> None:
        """Extract a single feature from a datetime column.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the datetime column
        column : str
            The name of the datetime column
        feature : str
        """
        extractor = self.feature_extractors[feature]
        feature_column = f"{column}_{feature}"

        try:
            if feature in ["year", "dayofyear"]:
                df.loc[:, feature_column] = extractor(df[column]).astype("uint16")
            elif feature in ["month", "day", "hour", "minute", "second", "weekday"]:
                df.loc[:, feature_column] = extractor(df[column]).astype("uint8")
            else:
                error_message = f"Unsupported feature: {feature}"
                raise ValueError(error_message)
        except AttributeError as exc:
            error_message = (
                f"Column '{column}' contains invalid datetime values. Please ensure that the column"
                " contains valid datetime values before extracting features."
            )
            raise ValueError(error_message) from exc

    def _drop_datetime_columns(self, df: pd.DataFrame, log: bool) -> pd.DataFrame:
        """Drop the datetime columns from the `df`."""
        if self.datetime_columns:
            if log:
                self.logger.info(f"Dropping original datetime columns: {self.datetime_columns}")
            return df.drop(columns=self.datetime_columns)
        return df

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
        self.logger.info("Calculating features")

        datasets = [
            ("flow", data.flow, True),
            ("train", data.train, True),
            ("validation", data.validation, False),
            ("test", data.test, False),
        ]

        for attr_name, dataset, should_log in datasets:
            if dataset is not None:
                ds = self._create_datetime_features(dataset, log=should_log)
                ds = self._drop_datetime_columns(ds, log=should_log)
                setattr(data, attr_name, ds)

        return data

    def _create_datetime_features(
        self, dataset: pd.DataFrame, *, log: Optional[bool] = False
    ) -> pd.DataFrame:
        """Create datetime features.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the datetime columns
        log : Optional[bool], optional
            Whether to log warnings and errors, by default False

        Returns
        -------
        pd.DataFrame
            The DataFrame with the datetime features added
        """
        created_features = []

        if self.datetime_columns:
            for column in self.datetime_columns:
                if column in dataset.columns:
                    dataset = self._convert_column_to_datetime(dataset, column, log)

                    if self.features:
                        for feature in self.features:
                            self._extract_feature(dataset, column, feature)
                            created_features.append(f"{column}_{feature}")
                    elif log:
                        self.logger.warning(
                            "No datetime features specified. Skipping feature extraction."
                        )
                elif log:
                    self.logger.warning("Datetime column '{column}' not found in the DataFrame")
        elif log:
            self.logger.warning("No datetime columns specified. Skipping feature extraction.")

        if log:
            self.logger.info(f"Created new features: {self.features}")

        return dataset
