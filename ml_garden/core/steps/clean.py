from typing import Optional

import pandas as pd

from ml_garden.core import DataContainer
from ml_garden.core.steps.base import PipelineStep


class CleanStep(PipelineStep):
    """Clean tabular data."""

    used_for_prediction = True
    used_for_training = True

    def __init__(
        self,
        fill_missing: Optional[dict] = None,
        remove_outliers: Optional[dict] = None,
        convert_dtypes: Optional[dict] = None,
        drop_na_columns: Optional[list] = None,
        drop_ids: Optional[dict] = None,
        filter_conditions: Optional[dict] = None,
    ) -> None:
        """Initialize CleanStep.

        Parameters
        ----------
        fill_missing : Optional[dict], optional
            Dictionary containing column names and fill values, by default None
        remove_outliers : Optional[dict], optional
            Dictionary containing column names and outlier removal methods, by default None
        convert_dtypes : Optional[dict], optional
            Dictionary containing column names and data types, by default None
        drop_na_columns : Optional[list], optional
            List of column names to drop rows with missing values, by default None
        drop_ids : Optional[dict], optional
            Dictionary containing column names and IDs to drop, by default None
        filter_conditions : Optional[dict], optional
            Dictionary containing column names and filter conditions, by default None
        """
        self.init_logger()
        self.fill_missing = fill_missing
        self.remove_outliers = remove_outliers
        self.convert_dtypes = convert_dtypes
        self.drop_na_columns = drop_na_columns
        self.drop_ids = drop_ids
        self.filter = filter_conditions

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
        self.logger.info("Cleaning tabular data...")

        if not data.is_train:
            data.flow = self._clean_df(data.flow)
            return data

        if data.train is not None:
            data.train = self._clean_df(data.train)

        if data.validation is not None:
            data.validation = self._clean_df(data.validation)

        if data.test is not None:
            data.test = self._clean_df(data.test)

        return data

    def _clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to clean

        Returns
        -------
        pd.DataFrame
            The cleaned DataFrame
        """
        df = self._filter(df)

        df = self._remove_outliers(df)

        df = self._fill_missing(df)

        df = self._convert_dtypes(df)

        df = self._drop_na_columns(df)

        df = self._drop_ids(df)

        return df

    def _filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to filter

        Returns
        -------
        pd.DataFrame
            The filtered DataFrame
        """
        if self.filter:
            original_rows = len(df)
            for key, value in self.filter.items():
                before_filter_rows = len(df)
                df = df.query(value)
                dropped_rows = before_filter_rows - len(df)
                dropped_percentage = (dropped_rows / before_filter_rows) * 100
                self.logger.info(
                    "Filter '%s': %s | Dropped rows: %d (%.2f%%)",
                    key,
                    value,
                    dropped_rows,
                    dropped_percentage,
                )
            total_dropped_rows = original_rows - len(df)
            total_dropped_percentage = (total_dropped_rows / original_rows) * 100
            self.logger.info(
                "Total rows dropped: %d (%.2f%%) | Final number of rows: %d",
                total_dropped_rows,
                total_dropped_percentage,
                len(df),
            )
        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to remove outliers from

        Returns
        -------
        pd.DataFrame
            The DataFrame without outliers
        """
        if self.remove_outliers:
            for column, method in self.remove_outliers.items():
                if column in df.columns:
                    if method == "clip":
                        q1 = df[column].quantile(0.25)
                        q3 = df[column].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - (1.5 * iqr)
                        upper_bound = q3 + (1.5 * iqr)
                        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
                        self.logger.info("Clipped outliers in column '%s'", column)
                    elif method == "drop":
                        q1 = df[column].quantile(0.25)
                        q3 = df[column].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - (1.5 * iqr)
                        upper_bound = q3 + (1.5 * iqr)
                        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
                        df = df[~outliers]
                        self.logger.info("Dropped outliers in column '%s'", column)
                    else:
                        self.logger.warning("Unsupported outlier removal method '%s'", method)
                else:
                    self.logger.warning("Column '%s' not found in the DataFrame", column)
        return df

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to fill missing values in

        Returns
        -------
        pd.DataFrame
            The DataFrame with missing values filled
        """
        if self.fill_missing:
            for column, fill_value in self.fill_missing.items():
                if column in df.columns:
                    df[column] = df[column].fillna(fill_value)
                    self.logger.info(
                        "Filled missing values in column '%s' with %s",
                        column,
                        fill_value,
                    )
                else:
                    self.logger.warning("Column '%s' not found in the DataFrame", column)
        return df

    def _convert_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert column data types in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to convert column data types in

        Returns
        -------
        pd.DataFrame
            The DataFrame with converted column data types
        """
        if self.convert_dtypes:
            for column, dtype in self.convert_dtypes.items():
                if column in df.columns:
                    df[column] = df[column].astype(dtype)
                    self.logger.info("Converted column '%s' to %s", column, dtype)
                else:
                    self.logger.warning("Column '%s' not found in the DataFrame", column)
        return df

    def _drop_na_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with missing values in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to drop rows with missing values in

        Returns
        -------
        pd.DataFrame
            The DataFrame without rows with missing values
        """
        if self.drop_na_columns:
            for column in self.drop_na_columns:
                if column in df.columns:
                    initial_rows = len(df)
                    df = df.dropna(subset=[column])
                    dropped_rows = initial_rows - len(df)
                    self.logger.info(
                        "Dropped %d rows with None values in column '%s'",
                        dropped_rows,
                        column,
                    )
                else:
                    self.logger.warning("Column '%s' not found in the DataFrame", column)
        return df

    def _drop_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with specific IDs in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to drop rows with specific IDs in

        Returns
        -------
        pd.DataFrame
            The DataFrame without rows with specific IDs
        """
        if self.drop_ids:
            for column, ids in self.drop_ids.items():
                if column in df.columns:
                    initial_rows = len(df)
                    initial_ids = set(df[column].unique())

                    dropped_ids = set(ids) & initial_ids
                    not_found_ids = set(ids) - initial_ids

                    if dropped_ids:
                        df = df.loc[~df[column].isin(dropped_ids)].copy()
                        dropped_rows = initial_rows - len(df)
                        percentage_dropped = (
                            dropped_rows / initial_rows
                        ) * 100  # Calculate the percentage of rows dropped
                        self.logger.info(
                            "Dropped %d rows (%.2f%%) with IDs %s in column '%s'",
                            dropped_rows,
                            percentage_dropped,
                            list(dropped_ids),
                            column,
                        )
                    else:
                        self.logger.info(
                            "No rows dropped for IDs %s in column '%s'",
                            list(ids),
                            column,
                        )

                    if not_found_ids:
                        self.logger.warning(
                            "IDs %s not found in column '%s'",
                            list(not_found_ids),
                            column,
                        )
                else:
                    self.logger.warning("Column '%s' not found in the DataFrame", column)
        return df
