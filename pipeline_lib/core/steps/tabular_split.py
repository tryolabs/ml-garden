from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline_lib.core import DataContainer

from .base import PipelineStep


class TabularSplitStep(PipelineStep):
    """Split the data."""

    def __init__(self, config: Optional[dict] = None) -> None:
        """Initialize SplitStep."""
        super().__init__(config=config)
        self.init_logger()

    def _id_based_split(
        self,
        df: pd.DataFrame,
        train_ids: list[str],
        validation_ids: list[str],
        id_column_name: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the DataFrame into training and validation sets based on specified IDs.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to split.
        train_ids : List[str]
            List of IDs for the training set.
        validation_ids : List[str]
            List of IDs for the validation set.
        id_column_name : str
            The name of the column in df that contains the IDs.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing the training set and the validation set.
        """
        train_df = df[df[id_column_name].isin(train_ids)]
        validation_df = df[df[id_column_name].isin(validation_ids)]
        return train_df, validation_df

    def _percentage_based_id_split(
        self, df: pd.DataFrame, train_percentage: float, id_column_name: str
    ) -> Tuple[list[str], list[str]]:
        """
        Splits the unique IDs into training and validation sets based on specified percentages.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the IDs.
        train_percentage : float
            The percentage of IDs to include in the training set.
        id_column_name : str
            The name of the column containing the IDs.

        Returns
        -------
        Tuple[List[str], List[str]]
            A tuple containing lists of training and validation IDs.
        """
        unique_ids = df[id_column_name].unique()
        train_ids, validation_ids = train_test_split(
            unique_ids, train_size=train_percentage, random_state=42
        )
        return train_ids.tolist(), validation_ids.tolist()

    def execute(self, data: DataContainer) -> DataContainer:
        """Execute the split based on IDs."""
        self.logger.info("Splitting tabular data...")

        split_configs = self.config

        if split_configs is None:
            self.logger.info("No split_configs found. No splitting will be performed.")
            return data

        df = data[DataContainer.CLEAN]
        id_column_name = split_configs.get("id_column")
        if not id_column_name:
            raise ValueError("ID column name must be specified in split_configs.")

        # check if both train_percentage and train_ids are provided
        if "train_percentage" in split_configs and "train_ids" in split_configs:
            raise ValueError(
                "Both train_percentage and train_ids cannot be provided in split_configs."
            )

        # check if either train_percentage or train_ids are provided
        if "train_percentage" not in split_configs and "train_ids" not in split_configs:
            raise ValueError(
                "Either train_percentage or train_ids must be provided in split_configs."
            )

        if "train_percentage" in split_configs:
            train_percentage = split_configs.get("train_percentage")
            if train_percentage is None or train_percentage <= 0 or train_percentage >= 1:
                raise ValueError("train_percentage must be between 0 and 1.")
            train_ids, validation_ids = self._percentage_based_id_split(
                df, train_percentage, id_column_name
            )
        else:
            train_ids = split_configs.get("train_ids")
            validation_ids = split_configs.get("validation_ids")
            if not train_ids or not validation_ids:
                raise ValueError(
                    "Both train_ids and validation_ids must be provided in split_configs unless"
                    " train_percentage is specified."
                )

        self.logger.info(f"Number of train ids: {len(train_ids)}")
        self.logger.info(f"Number of validation ids: {len(validation_ids)}")

        train_df, validation_df = self._id_based_split(
            df, train_ids, validation_ids, id_column_name
        )

        train_rows = len(train_df)
        validation_rows = len(validation_df)
        total_rows = train_rows + validation_rows

        self.logger.info(
            f"Number of rows in training set: {len(train_df)} | {train_rows/total_rows:.2%}"
        )
        self.logger.info(
            f"Number of rows in validation set: {len(validation_df)} |"
            f" {validation_rows/total_rows:.2%}"
        )

        left_ids = df[~df[id_column_name].isin(train_ids + validation_ids)][id_column_name].unique()
        self.logger.info(f"Number of IDs left from total df: {len(left_ids)}")
        self.logger.debug(f"IDs left from total df: {left_ids}")

        data[DataContainer.TRAIN] = train_df
        data[DataContainer.VALIDATION] = validation_df

        return data
