from math import isclose
from typing import Optional

from sklearn.model_selection import train_test_split

from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps.base import PipelineStep
from pipeline_lib.utils.string_utils import concatenate_columns


class TabularSplitStep(PipelineStep):
    """Split the data."""

    used_for_prediction = False
    used_for_training = True

    def __init__(
        self,
        train_percentage: float,
        validation_percentage: Optional[float] = None,
        test_percentage: Optional[float] = None,
        group_by_columns: Optional[list[str]] = None,
    ) -> None:
        """Initialize SplitStep."""
        self.init_logger()
        self.train_percentage = train_percentage
        self.validation_percentage = validation_percentage
        self.test_percentage = test_percentage
        self.group_by_columns = group_by_columns

        if self.train_percentage <= 0 or self.train_percentage >= 1:
            raise ValueError("train_percentage must be between 0 and 1.")

        if self.validation_percentage is not None:
            if self.validation_percentage <= 0 or self.validation_percentage >= 1:
                raise ValueError("validation_percentage must be between 0 and 1.")
            if self.test_percentage is None:
                if self.train_percentage + self.validation_percentage != 1:
                    raise ValueError(
                        "The sum of train_percentage and validation_percentage must equal 1 when"
                        " test_percentage is not specified."
                    )
            else:
                if not isclose(
                    self.train_percentage + self.validation_percentage + self.test_percentage, 1
                ):
                    raise ValueError(
                        "The sum of train_percentage, validation_percentage, and test_percentage"
                        " must equal 1."
                    )

        if self.test_percentage is not None:
            if self.test_percentage <= 0 or self.test_percentage >= 1:
                raise ValueError("test_percentage must be between 0 and 1.")
            if self.validation_percentage is None:
                raise ValueError(
                    "validation_percentage must be provided when test_percentage is specified."
                )

    def execute(self, data: DataContainer) -> DataContainer:
        """Execute the random train-validation-test split."""
        self.logger.info("Splitting tabular data...")
        df = data.flow

        if self.test_percentage is not None and data.test is not None:
            raise ValueError(
                "Cannot set test_percentage in TabularSplitStep when data.test is already set. "
                "If the test dataset was provided in the generate step, do not set test_percentage."
            )

        if self.group_by_columns is not None:
            concatted_groupby_columns = concatenate_columns(df, self.group_by_columns)
            split_values = concatted_groupby_columns.unique().tolist()
        else:
            concatted_groupby_columns = None
            split_values = df.index.tolist()

        if self.test_percentage is not None:
            train_val_values, test_values = train_test_split(
                split_values, test_size=self.test_percentage, random_state=42
            )
            train_values, validation_values = train_test_split(
                train_val_values,
                train_size=self.train_percentage
                / (self.train_percentage + self.validation_percentage),
                random_state=42,
            )
        else:
            train_values, validation_values = train_test_split(
                split_values, train_size=self.train_percentage, random_state=42
            )
            test_values = None

        if self.group_by_columns is not None:
            train_df = df[concatted_groupby_columns.isin(set(train_values))].copy()
            validation_df = df[concatted_groupby_columns.isin(set(validation_values))].copy()

            if test_values:
                test_df = df[concatted_groupby_columns.isin(set(test_values))].copy()
            else:
                test_df = None
        else:
            train_df = df[df.index.isin(set(train_values))].copy()
            validation_df = df[df.index.isin(set(validation_values))].copy()
            if test_values:
                test_df = df[df.index.isin(set(test_values))].copy()

        data.train = train_df
        data.validation = validation_df
        if test_values is not None:
            data.test = test_df

        # Logging
        if self.group_by_columns is not None:
            train_groups = len(train_values)
            validation_groups = len(validation_values)
            test_groups = len(test_values) if test_values is not None else 0
            total_groups = train_groups + validation_groups + test_groups

            self.logger.info(f"Using group by columns for splits based on: {self.group_by_columns}")
            self.logger.info(
                f"Number of groups in train set: {train_groups} | {train_groups / total_groups:.2%}"
            )
            self.logger.info(
                f"Number of groups in validation set: {validation_groups} |"
                f" {validation_groups / total_groups:.2%}"
            )
            if test_values is not None:
                self.logger.info(
                    f"Number of groups in test set: {test_groups} |"
                    f" {test_groups / total_groups:.2%}"
                )

        train_rows = len(data.train)
        validation_rows = len(data.validation)
        test_rows = len(data.test) if data.test is not None else 0

        total_rows = train_rows + validation_rows + test_rows

        self.logger.info(
            f"Number of rows in training set: {train_rows} | {train_rows / total_rows:.2%}"
        )
        self.logger.info(
            f"Number of rows in validation set: {validation_rows} |"
            f" {validation_rows / total_rows:.2%}"
        )
        if test_values is not None or data.test is not None:
            self.logger.info(
                f"Number of rows in test set: {test_rows} | {test_rows / total_rows:.2%}"
            )

        return data
