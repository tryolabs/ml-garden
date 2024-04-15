from typing import Iterable, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline_lib.core import DataContainer
from pipeline_lib.core.random_state_generator import get_random_state
from pipeline_lib.core.steps.base import PipelineStep


def _concatenate_columns(
    df: pd.DataFrame,
    cols: Iterable[str],
    sep: str = " ",
    na_rep: str = "",
) -> pd.Series:
    """
    Concatenates the specified `cols` of a DataFrame into a single column.

    This function fills NaN values with a specified string, converts the columns to string type,
    concatenates them with a specified separator and returns the concatenated values.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to process.
    cols : iterable of str or None, optional
        An iterable of column names to concatenate. If None, will choose all columns like `into_col`
        with `df.filter`. Default is None.
    sep : str, optional
        The separator to use when concatenating. Default is " ".
    na_rep : str, optional
        The string to use to replace NaN values. Default is "".

    Returns
    -------
    pandas.DataFrame
        The processed DataFrame, which includes the new column and excludes the original columns.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'A': ['a', 'b', np.nan],
    ...     'B': ['c', np.nan, 'd'],
    ...     'C': ['e', 'f', 'g']
    ... })
    >>> concatenate_columns(df, ['A', 'B', 'C'])
       D
    0  a c e
    1  b f
    2  d g
    """
    to_concat = df[cols].fillna(na_rep).astype(str)
    concatted = to_concat.iloc[:, 0].str.cat(others=to_concat.iloc[:, 1:], sep=sep)

    return concatted


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
        random_seed: Optional[int] = 42,
    ) -> None:
        """Initialize SplitStep."""
        self.init_logger()
        self.train_percentage = train_percentage
        self.validation_percentage = validation_percentage
        self.test_percentage = test_percentage
        self.group_by_columns = group_by_columns
        self.random_seed = random_seed

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

    def perform_split(
        self,
        df: pd.DataFrame,
        data: DataContainer,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Perform the split that was fit during pipeline training on a potentially new dataframe.

        During pipeline training, this step will store in the DataContainer necessary information to
        determine which set (train/val/test) a row belongs to.
        - If using simple random splitting, it will store the postprocessed df indices
        - If using group based splitting, it will store the group columns' concattenated values, ie
            - if groups are identified by the id, timestamp columns, the DataContainer will store:
                - "train": [<id_train_1_timestamp_train_1>, <id_train_1_timestamp_train_2>, ...]
                - "validation": [<id_validation_1_timestamp_validation_1>, ...]
                - "test": [<id_test_1_timestamp_test_1>, <id_test_1_timestamp_test_2>, ...]

        These values will be compared to the rows' values to determine the set the row belongs to.
        For example in the below df:
            row         id              timestamp             ...
            0           id_train_1      timestamp_train_1     ...
            1           id_train_1      timestamp_train_2      ...
            2           id_validation_1  timestamp_validation_1 ...
            3           id_test_1       timestamp_test_1        ...

        running this method would result in 3 dataframes:
        - train, having rows 0 and 1
        - validation, having row w
        - test, having row 3

        Args:
            df (pd.DataFrame): the dataframe to be split
            data (DataContainer): the training DataContainer object, with the stored split values

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]: _description_


        Example usage:
        # %%
        from pipeline_lib import Pipeline

        # Load a trained pipeline from it's config
        pipeline = Pipeline.from_json("example.json")

        # Retrieve the split step from the pipeline
        split_step: TabularSplitStep = [
            step for step in pipeline.steps if isinstance(step, TabularSplitStep)
        ][0]

        # Perform the split
        train, val, test = split_step.perform_split(data.raw, data)
        """

        train_values, validation_values, test_values = (
            data.split_values["train"],
            data.split_values["validation"],
            data.split_values["test"],
        )

        if self.group_by_columns is not None:
            # Group based split
            concatted_groupby_columns = _concatenate_columns(df, self.group_by_columns)
            train_df = df[concatted_groupby_columns.isin(set(train_values))]
            validation_df = df[concatted_groupby_columns.isin(set(validation_values))]

            if test_values:
                test_df = df[concatted_groupby_columns.isin(set(test_values))]
            else:
                test_df = None
        else:
            # Simple random sampling based split
            train_df = df[df.index.isin(set(train_values))]
            validation_df = df[df.index.isin(set(validation_values))]
            if test_values:
                test_df = df[df.index.isin(set(test_values))]

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
            if test_df is not None:
                self.logger.info(
                    f"Number of groups in test set: {test_groups} |"
                    f" {test_groups / total_groups:.2%}"
                )

        train_rows = len(train_df)
        validation_rows = len(validation_df)
        test_rows = len(test_df) if test_df is not None else 0
        total_rows = train_rows + validation_rows + test_rows

        self.logger.info(
            f"Number of rows in training set: {train_rows} | {train_rows / total_rows:.2%}"
        )
        self.logger.info(
            f"Number of rows in validation set: {validation_rows} |"
            f" {validation_rows / total_rows:.2%}"
        )
        if test_df is not None:
            self.logger.info(
                f"Number of rows in test set: {test_rows} | {test_rows / total_rows:.2%}"
            )

        return train_df, validation_df, test_df

    def execute(self, data: DataContainer) -> DataContainer:
        """Execute the random train-validation-test split."""
        self.logger.info("Splitting tabular data...")
        df = data.flow

        if self.group_by_columns is not None:
            concatted_groupby_columns = _concatenate_columns(df, self.group_by_columns)
            split_values = concatted_groupby_columns.unique().tolist()
        else:
            concatted_groupby_columns = None
            split_values = df.index.tolist()

        if self.test_percentage is not None:
            train_val_split_values, test_split_values = train_test_split(
                split_values, test_size=self.test_percentage, random_state=42
            )
            train_split_values, validation_split_values = train_test_split(
                train_val_split_values,
                train_size=self.train_percentage
                / (self.train_percentage + self.validation_percentage),
                random_state=42,
            )
        else:
            train_split_values, validation_split_values = train_test_split(
                split_values, train_size=self.train_percentage, random_state=42
            )
            test_split_values = []

        data.split_values = {
            "train": train_split_values,
            "validation": validation_split_values,
            "test": test_split_values,
        }

        data.train, data.validation, data.test = self.perform_split(df, data)

        return data
