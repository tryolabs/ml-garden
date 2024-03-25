from typing import List, Optional

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps.base import PipelineStep


class UnsupportedFeatureError(Exception):
    """Custom exception for unsupported features."""

    pass


class CalculateFeaturesStep(PipelineStep):
    """Calculate features."""

    used_for_prediction = True
    used_for_training = True

    def __init__(
        self,
        datetime_columns: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
    ) -> None:
        """Initialize CalculateFeaturesStep."""
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
                raise UnsupportedFeatureError(
                    f"Unsupported datetime features: {unsupported_features}"
                )

    def _convert_column_to_datetime(self, df: pd.DataFrame, column: str) -> None:
        """Convert a column to datetime."""
        # Check if the column is already a datetime type
        if not is_datetime64_any_dtype(df[column]):
            try:
                df.loc[:, column] = pd.to_datetime(df[column], errors="raise")
                self.logger.info(f"Column '{column}' converted to datetime.")
            except Exception as e:
                self.logger.error(f"Error converting column '{column}' to datetime: {e}")
        else:
            self.logger.debug(f"Column '{column}' is already a datetime type.")

    def _extract_feature(self, df: pd.DataFrame, column: str, feature: str) -> None:
        """Extract a single feature from a datetime column."""
        extractor = self.feature_extractors[feature]
        df.loc[:, f"{column}_{feature}"] = extractor(df[column])

    def execute(self, data: DataContainer) -> DataContainer:
        """Execute the step."""
        self.logger.info("Calculating features")

        df = data.flow
        created_features = []

        if self.datetime_columns:
            for column in self.datetime_columns:
                if column in df.columns:
                    self._convert_column_to_datetime(df, column)
                    if self.features:
                        for feature in self.features:
                            self._extract_feature(df, column, feature)
                            created_features.append(f"{column}_{feature}")
                    else:
                        self.logger.warning(
                            "No datetime features specified. Skipping feature extraction."
                        )
                else:
                    self.logger.warning(f"Datetime column '{column}' not found in the DataFrame")
        else:
            self.logger.warning("No datetime columns specified. Skipping feature extraction.")

        # drop original datetime columns
        if self.datetime_columns:
            df = df.drop(columns=self.datetime_columns)
            self.logger.info(f"Dropped datetime columns: {self.datetime_columns}")

        self.logger.info(f"Created new features: {created_features}")

        data.flow = df

        return data
