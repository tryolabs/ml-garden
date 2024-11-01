import datetime
from collections.abc import Hashable
from contextlib import suppress
from decimal import Decimal
from typing import Optional, Type

import numpy as np
import pandas as pd

# ruff: noqa: C901 PLR0912

NOT_PRESENT_STRING = "NA"


def detect_categorical(
    column: pd.Series,
    q: float = 0.75,
    t: int = 1,
    considered_dtype_names: Optional[list[str]] = None,
    skip_cols: Optional[set[str]] = None,
) -> bool:
    if considered_dtype_names is None:
        considered_dtype_names = ["object", "string", "int", "int64", "category"]
    # Consider column as categorical if:
    # - it's dtype name is among the ones considered and
    # -- it already is a pandas categorical or
    # -- it's a hashable dtype and q*100% of the unique values in the column
    # -- occur more than t times. I.E: if 75% of unique values in column
    # -- occur more than once, then consider it a categorical
    if skip_cols is None:
        skip_cols = set()

    if column.dtype.name in considered_dtype_names and column.name not in skip_cols:
        if column.dtype.name == "category":
            return True
        else:
            idx = column.first_valid_index()
            if idx is not None:
                return (
                    isinstance(column.loc[idx], Hashable)
                    and np.quantile(a=column.value_counts(), q=q) > t
                )
            else:
                return False
    else:
        return False


def detect_categoricals(
    df: pd.DataFrame,
    q: float = 0.75,
    t: int = 1,
    considered_dtype_names: Optional[list[str]] = None,
    skip_cols: Optional[set[str]] = None,
) -> pd.Series:
    if considered_dtype_names is None:
        considered_dtype_names = ["object", "int", "int64", "category"]
    return df.apply(
        detect_categorical,
        q=q,
        t=t,
        considered_dtype_names=considered_dtype_names,
        skip_cols=skip_cols,
    )


def convert_to_categoricals(
    df: pd.DataFrame,
    q: float = 0.75,
    t: int = 1,
    considered_dtype_names: Optional[list[str]] = None,
    skip_cols: Optional[set[str]] = None,
    not_present_string: Optional[str] = NOT_PRESENT_STRING,
) -> pd.DataFrame:
    if considered_dtype_names is None:
        considered_dtype_names = ["object", "string"]
    # For conversion the default considered_dtype_names includes only object
    # columns since they are the ones that can get performance gains from
    # converting to categories
    categorical_features = detect_categoricals(
        df, q=q, t=t, considered_dtype_names=considered_dtype_names, skip_cols=skip_cols
    )
    for column in categorical_features.index[categorical_features]:
        df[column] = df[column].astype("category")
        if not_present_string is not None:
            # Create a new category value for not_present_string and fill NaNs in categorical data
            # with it
            if not_present_string not in df[column].cat.categories:
                df[column] = df[column].cat.add_categories(not_present_string)
            df[column] = df[column].fillna(not_present_string)

    return df


def type_of_first_non_na(s: pd.Series) -> Optional[Type]:
    first_non_na = s.first_valid_index()
    if first_non_na is None:
        return None
    if isinstance(first_non_na, pd.Series):
        return type(s[first_non_na.iloc[0]])
    else:
        return type(s[first_non_na])


def detect_string_columns(
    df: pd.DataFrame,
    skip_cols: Optional[set[str]] = None,
    *,
    object_cols_only: bool = False,
) -> list[str]:
    """
    Detect string columns and return a list of their names.

    This method examines the contents of "object" and "string" columns and if the first
    non-null value found is a string, then the column is considered a string column

    Args:
        df (pd.DataFrame): dataframe to perform the detection on
        skip_cols (_type_, optional): optional list of columns to be ignored.
            Defaults to set().
        object_cols_only (bool, optional): if true perform detection on "object" dtyped
            columns only, otherwise check "string" or "object" dtyped columns. Defaults to
            False.

    Returns
    -------
        list[str]: list of string column names
    """
    if skip_cols is None:
        skip_cols = []

    res = []
    dtypes = df.dtypes
    for column in dtypes.index:
        if column not in skip_cols:
            if (dtypes[column].name == "string") and not object_cols_only:
                res.append(column)
            elif dtypes[column].name == "object":
                idx_first_non_null = df[column].first_valid_index()
                if idx_first_non_null is not None and isinstance(
                    df.loc[idx_first_non_null, column], str
                ):
                    res.append(column)

    return res


def convert_object_columns_to_base_type(
    df: pd.DataFrame,
    skip_cols: Optional[set[str]] = None,
    *,
    convert_object_all_nans_to_float32: bool = True,
    convert_ints_with_nans_to_float32: bool = True,
    convert_bools_with_nans_to_float32: bool = True,
    fill_string_nans: bool | None = True,
) -> pd.DataFrame:
    if skip_cols is None:
        skip_cols = set()

    object_cols = (
        df.drop(columns=[col for col in skip_cols if col in df.columns]).dtypes == "object"
    )
    object_cols = object_cols.index[object_cols.to_numpy()].tolist()
    no_nan_object_columns = (~df.loc[:, object_cols].isna()).all(axis=0)
    no_nan_object_columns = no_nan_object_columns[no_nan_object_columns].index.tolist()

    # Interactive debugging assignments
    # col = no_nan_object_columns[0] # noqa: ERA001
    for col in no_nan_object_columns:
        first_non_na_type = type_of_first_non_na(df[col])
        if (first_non_na_type in (datetime.date, datetime.datetime)) or ("datetime64") in str(
            first_non_na_type
        ):
            df[col] = pd.to_datetime(df[col])
        else:
            # Savage
            # Try every possible dtype in increasing order of complexity
            # Didn't find any better solution.
            # I can speak for hours as to why Pandas doesn't provide =)
            try:
                df[col] = pd.to_numeric(df[col].values, downcast="unsigned")
            except ValueError:
                try:
                    df[col] = pd.to_numeric(df[col].values, downcast="integer")
                except ValueError:
                    try:
                        df[col] = pd.to_numeric(df[col].values, downcast="float")
                    except ValueError:
                        df[col] = df[col].astype(pd.StringDtype())
            except TypeError:
                # True objects can't be converted
                pass

    # col = 'is_qced' # noqa: ERA001
    any_nan_object_columns = (df.loc[:, object_cols].isna()).any(axis=0)
    any_nan_object_columns = any_nan_object_columns[any_nan_object_columns].index
    for col in any_nan_object_columns:
        first_non_na_type = type_of_first_non_na(s=df[col])

        if first_non_na_type == str:
            df[col] = df[col].astype(pd.StringDtype())
            continue

        is_float_to_be_converted = first_non_na_type in [float, Decimal]
        is_int_to_be_converted = convert_ints_with_nans_to_float32 and (
            first_non_na_type in [int, np.int64]
        )
        is_object_to_be_converted = convert_object_all_nans_to_float32 and (
            first_non_na_type is None
        )
        is_bool_to_be_converted = convert_bools_with_nans_to_float32 and (first_non_na_type == bool)

        do_float_conversion = (
            is_float_to_be_converted
            or is_int_to_be_converted
            or is_object_to_be_converted
            or is_bool_to_be_converted
        )
        if do_float_conversion:
            with suppress(ValueError):
                df[col] = pd.to_numeric(df[col].values, downcast="float")
            continue

    if fill_string_nans:
        string_columns = detect_string_columns(df, skip_cols=skip_cols, object_cols_only=True)
        if string_columns:
            df.loc[:, string_columns].fillna("")
            df.loc[:, string_columns] = df.loc[:, string_columns].astype(pd.StringDtype())
    return df


def downcast_int64_and_float64(
    df: pd.DataFrame, skip_cols: Optional[set[str]] = None, *, use_unsigned: bool = True
) -> pd.DataFrame:
    if skip_cols is None:
        skip_cols = set()

    dtypes = df.dtypes
    for column in dtypes.index:
        if column not in skip_cols:
            col_dtype_name = dtypes[column].name.lower()
            if col_dtype_name in ["int32", "int64"]:
                if use_unsigned:
                    do_unsigned = (~df[column].isna()).any() and (df[column].min() >= 0)
                else:
                    do_unsigned = False

                if do_unsigned:
                    df[column] = pd.to_numeric(df[column].values, downcast="unsigned")
                elif col_dtype_name == "int64":
                    df[column] = pd.to_numeric(df[column].values, downcast="integer")
            elif col_dtype_name == "float64":
                df[column] = pd.to_numeric(df[column].values, downcast="float")
    return df


def apply_all_dtype_conversions(
    df: pd.DataFrame,
    skip_cols: Optional[set[str]] = None,
    not_present_string: Optional[str] = NOT_PRESENT_STRING,
) -> pd.DataFrame:
    """Apply all dtype conversion optimizations.

    Converts repeating value columns to categoricals, object typed columns to the base type of their
      values (if possible, if values share a single datatype), and downcasts integers and floats to
      the minimum required precision.
    Optimizations will be performed in place to save on RAM.

    Args:
        df (pd.DataFrame): dataframe to apply the optimizations to
        skip_cols (Optional[set[str]], optional): columns to skip from the optimization. These will
          remain untouched. Defaults to None.
        not_present_string (Optional[str], optional): string value to use for representing missing
          values in categorical columns. After conversion, all categorical columns will be checked.
          Those not having not_present_string in their values will have it added to their
          categories. After adding to the categories missing values will be filled with this string
          to avoid having nans in the column (and hence needing an object column dtype). If set to
          None this process will be skipped. Defaults to NOT_PRESENT_STRING.

    Returns
    -------
        pd.DataFrame: _description_
    """
    df = convert_to_categoricals(df, skip_cols=skip_cols, not_present_string=not_present_string)
    df = convert_object_columns_to_base_type(df, skip_cols=skip_cols)
    df = downcast_int64_and_float64(df, skip_cols=skip_cols)
    return df


def calculate_memory_storage(df: pd.DataFrame, base: int = 2**20, round_units: int = 2) -> float:
    return round(df.memory_usage(deep=True).sum() / base, round_units)
