from typing import Optional

from sklearn.model_selection import train_test_split

from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps.base import PipelineStep


class TabularSplitStep(PipelineStep):
    """Split the data."""

    used_for_prediction = False
    used_for_training = True

    def __init__(
        self,
        train_percentage: float,
        validation_percentage: Optional[float] = None,
        test_percentage: Optional[float] = None,
    ) -> None:
        """Initialize SplitStep."""
        self.init_logger()
        self.train_percentage = train_percentage
        self.validation_percentage = validation_percentage
        self.test_percentage = test_percentage

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
                if self.train_percentage + self.validation_percentage + self.test_percentage != 1:
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

        if self.test_percentage is not None:
            train_val_df, test_df = train_test_split(
                df, test_size=self.test_percentage, random_state=42
            )
            train_df, validation_df = train_test_split(
                train_val_df,
                train_size=self.train_percentage
                / (self.train_percentage + self.validation_percentage),
                random_state=42,
            )
        else:
            train_df, validation_df = train_test_split(
                df, train_size=self.train_percentage, random_state=42
            )
            test_df = None

        train_rows = len(train_df)
        validation_rows = len(validation_df)
        test_rows = len(test_df) if test_df is not None else 0
        total_rows = train_rows + validation_rows + test_rows

        self.logger.info(
            f"Number of rows in training set: {train_rows} | {train_rows/total_rows:.2%}"
        )
        self.logger.info(
            f"Number of rows in validation set: {validation_rows} |"
            f" {validation_rows/total_rows:.2%}"
        )
        if test_df is not None:
            self.logger.info(
                f"Number of rows in test set: {test_rows} | {test_rows/total_rows:.2%}"
            )

        data.train = train_df
        data.validation = validation_df
        if test_df is not None:
            data.test = test_df

        return data
