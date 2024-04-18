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
        """
        Initialize the TabularSplitStep

        Parameters
        ----------
        train_percentage : float
            The percentage of data to use for the training set.
        validation_percentage : Optional[float], optional
            The percentage of data to use for the validation set. If not provided it will default to
            1 - (validation_percentage - test_percentage), by default None
        test_percentage : Optional[float], optional
            The percentage of data to use for the test set. If not provided it will default to no
            test set, by default None
        group_by_columns : Optional[list[str]], optional
            Columns defining the groups by which the splits will be performed, by default None
            Useful for time series data. By providing the series identifiers in this parameter,
            the splits will be performed so that entire series are not split across different splits

        Raises
        ------
        ValueError
            In case the parameters are invalid
        """
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

    def execute(self, data: DataContainer) -> DataContainer:
        """
        Execute the random train-validation-test split.

        After training, users can obtain the train/validation/test sets from a stored DataContainer
        with the following code:

        train, validation, test = (
            df.loc[data.split_indices["train"]],
            df.loc[data.split_indices["validation"]],
            df.loc[data.split_indices["test"]] if len(data.split_indices["test"]) > 0 else None,
        )

        Where df is the DataFrame used as input to the SplitStep
        """

        self.logger.info("Splitting tabular data...")
        df = data.flow

        if self.group_by_columns is not None:
            # Group based splits
            concatted_groupby_columns = _concatenate_columns(df, self.group_by_columns)
            split_values = concatted_groupby_columns.unique().tolist()
        else:
            # Simple Random Split
            concatted_groupby_columns = None
            split_values = df.index.tolist()

        if self.test_percentage is not None:
            # Train, Validation and Test split
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
            # Train and Validation split only
            train_split_values, validation_split_values = train_test_split(
                split_values, train_size=self.train_percentage, random_state=42
            )
            test_split_values = []

        if self.group_by_columns is not None:
            # Group based splits
            train_split_indices = df[concatted_groupby_columns.isin(set(train_split_values))].index
            validation_split_indices = df[
                concatted_groupby_columns.isin(set(validation_split_values))
            ].index
            test_split_indices = (
                df[concatted_groupby_columns.isin(set(test_split_values))].index
                if len(test_split_values) > 0
                else pd.Index(data=[], dtype=df.index.dtype)
            )
        else:
            # Simple Random Split
            train_split_indices = df[df.index.isin(set(train_split_values))].index
            validation_split_indices = df[df.index.isin(set(validation_split_values))].index
            test_split_indices = (
                df[df.index.isin(set(test_split_values))].index
                if len(test_split_values) > 0
                else pd.Index(data=[], dtype=df.index.dtype)
            )

        # Store the split indices in the DataContainer for later use
        data.split_indices = {
            "train": train_split_indices,
            "validation": validation_split_indices,
            "test": test_split_indices,
        }

        # Perform the actual split of the DataFrame into train/validation/test sets and store them
        # in the DataContainer
        data.train, data.validation, data.test = (
            df.loc[data.split_indices["train"]],
            df.loc[data.split_indices["validation"]],
            df.loc[data.split_indices["test"]] if len(data.split_indices["test"]) > 0 else None,
        )

        return data
