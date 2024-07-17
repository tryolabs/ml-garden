from math import isclose
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_garden.core import DataContainer
from ml_garden.core.random_state_generator import get_random_state
from ml_garden.core.steps.base import PipelineStep
from ml_garden.utils.string_utils import concatenate_columns


class TabularSplitStep(PipelineStep):
    """Split the data."""

    used_for_prediction = True
    used_for_training = True

    def __init__(
        self,
        train_percentage: float,
        validation_percentage: Optional[float] = None,
        test_percentage: Optional[float] = None,
        group_by_columns: Optional[list[str]] = None,
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

    def _set_X_y(self, data: DataContainer) -> None:
        """
        Set X and y for train, validation, and test sets.

        This method sets X_train, y_train, X_validation, y_validation, X_test, and y_test
        in the DataContainer object.

        Parameters
        ----------
        data : DataContainer
            The DataContainer object containing the split data.

        Returns
        -------
        None
        """

        def split_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
            df = df.drop(columns=data.columns_to_ignore_for_training)
            if data.target is None:
                return df, None
            return df.drop(columns=[data.target]), df[data.target]

        data.X_train, data.y_train = split_X_y(data.train)
        data.X_validation, data.y_validation = split_X_y(data.validation)

        if data.test is not None:
            data.X_test, data.y_test = split_X_y(data.test)
        else:
            data.X_test, data.y_test = None, None

        self.logger.debug("X and y sets have been created for train, validation, and test data.")

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
        if not data.is_train:
            data.X_prediction = data.flow
            if data.columns_to_ignore_for_training:
                data.X_prediction = data.X_prediction.drop(
                    columns=data.columns_to_ignore_for_training
                )
            return data

        self.logger.info("Splitting tabular data...")
        df = data.flow

        # Since we'll be using the dataframe indices for splitting, we need to make sure there are
        # no duplicates, otherwise slicing may produce more rows than expected.
        # For example if you have:
        # df=
        # idx,  A,  B
        # 0     a   b
        # 0     b   c
        # 1     d   e
        # and do df.loc[0] you get:
        # idx,  A,  B
        # 0      a   b
        # 0      b   c
        # But if you do df.loc[1] you get:
        # pd.Series(A=d, B=e)
        # Return types are different, since one returns a DataFrame and the other a Series, which
        # may break downstream code.
        # Also if idx = pd.Index(data=[0]), len(idx)==1 but len(df.loc[idx])==2, which can cause
        # inconsistencies and make interpretation difficult to understand
        assert not df.index.duplicated().any(), (
            "Duplicate indices found in the dataframe before split. Please ensure dataframe indices"
            "are unique before feeding them to the SplitStep."
        )

        if self.test_percentage is not None and data.test is not None:
            raise ValueError(
                "Cannot set test_percentage in TabularSplitStep when data.test is already set. "
                "If the test dataset was provided in the generate step, do not set test_percentage."
            )

        if self.group_by_columns is not None:
            # Group based splits
            concatted_groupby_columns = concatenate_columns(df, self.group_by_columns)
            split_values = concatted_groupby_columns.unique().tolist()
        else:
            # Simple Random Split
            concatted_groupby_columns = None
            split_values = df.index.tolist()

        if self.test_percentage is not None:
            # Train, Validation and Test split
            train_val_split_values, test_split_values = train_test_split(
                split_values, test_size=self.test_percentage, random_state=get_random_state()
            )
            train_split_values, validation_split_values = train_test_split(
                train_val_split_values,
                train_size=self.train_percentage
                / (self.train_percentage + self.validation_percentage),
                random_state=get_random_state(),
            )
        else:
            # Train and Validation split only
            train_split_values, validation_split_values = train_test_split(
                split_values, train_size=self.train_percentage, random_state=get_random_state()
            )

            test_split_values = []

        if concatted_groupby_columns is not None:
            # Group based splits
            train_split_indices = df[concatted_groupby_columns.isin(set(train_split_values))].index
            validation_split_indices = df[
                concatted_groupby_columns.isin(set(validation_split_values))
            ].index

            # Test dataset not provided, generate as a regular split
            test_split_indices = (
                df[concatted_groupby_columns.isin(set(test_split_values))].index
                if len(test_split_values) > 0
                else pd.Index(data=[], dtype=df.index.dtype)
            )
        else:
            # Simple Random Split
            train_split_indices = df[df.index.isin(set(train_split_values))].index
            validation_split_indices = df[df.index.isin(set(validation_split_values))].index

            # Test dataset not provided, generate as a regular split
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
        data.train, data.validation = (
            df.loc[data.split_indices["train"]],
            df.loc[data.split_indices["validation"]],
        )
        if data.test is None:
            # Test dataset not provided, generate as a regular split
            data.test = (
                df.loc[data.split_indices["test"]] if len(data.split_indices["test"]) > 0 else None
            )

        # Logging
        if self.group_by_columns is not None:
            train_groups = len(train_split_values)
            validation_groups = len(validation_split_values)
            test_groups = len(test_split_values)
            total_groups = train_groups + validation_groups + test_groups

            self.logger.info(f"Using group by columns for splits based on: {self.group_by_columns}")
            self.logger.info(
                f"Number of groups in train set: {train_groups} | {train_groups / total_groups:.2%}"
            )
            self.logger.info(
                f"Number of groups in validation set: {validation_groups} |"
                f" {validation_groups / total_groups:.2%}"
            )
            if data.test is not None:
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
        if data.test is not None or data.test is not None:
            self.logger.info(
                f"Number of rows in test set: {test_rows} | {test_rows / total_rows:.2%}"
            )

        # Set X and y for train, validation and test datasets
        self._set_X_y(data)

        return data
