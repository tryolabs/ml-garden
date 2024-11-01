"""Calculate datetime-related features from specified columns."""

from typing import List, Optional, Union

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
        datetime_columns: Optional[Union[List[str], str]] = None,
        features: Optional[List[str]] = None,
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

    def _convert_column_to_datetime(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Convert a column to datetime.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the column to convert
        column : str
            The name of the column to convert

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
                self.logger.info("Column '%s' automatically converted to datetime.", column)
            except Exception:
                self.logger.exception("Error converting column '%s' to datetime.", column)
        else:
            self.logger.debug("Column '%s' is already a datetime type.", column)

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

        if not data.is_train:
            data.flow = self._create_datetime_features(data.flow, log=True)

        if data.train is not None:
            data.train = self._create_datetime_features(data.train, log=True)

        if data.validation is not None:
            data.validation = self._create_datetime_features(data.validation)

        if data.test is not None:
            data.test = self._create_datetime_features(data.test)

        ## add datetime columns to ignore columns for training
        if self.datetime_columns:
            data.columns_to_ignore_for_training.extend(self.datetime_columns)

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
                    dataset = self._convert_column_to_datetime(dataset, column)

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

        return dataset
