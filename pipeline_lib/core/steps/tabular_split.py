from sklearn.model_selection import train_test_split

from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps.base import PipelineStep


class TabularSplitStep(PipelineStep):
    """Split the data."""

    used_for_prediction = False
    used_for_training = True

    def __init__(self, train_percentage: float) -> None:
        """Initialize SplitStep."""
        self.init_logger()
        self.train_percentage = train_percentage

        if self.train_percentage <= 0 or self.train_percentage >= 1:
            raise ValueError("train_percentage must be between 0 and 1.")

    def execute(self, data: DataContainer) -> DataContainer:
        """Execute the random train-validation split."""
        self.logger.info("Splitting tabular data...")

        df = data.flow

        train_df, validation_df = train_test_split(
            df, train_size=self.train_percentage, random_state=42
        )

        train_rows = len(train_df)
        validation_rows = len(validation_df)
        total_rows = train_rows + validation_rows

        self.logger.info(
            f"Number of rows in training set: {train_rows} | {train_rows/total_rows:.2%}"
        )
        self.logger.info(
            f"Number of rows in validation set: {validation_rows} |"
            f" {validation_rows/total_rows:.2%}"
        )

        data.train = train_df
        data.validation = validation_df

        return data
