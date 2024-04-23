from typing import Iterable

import pandas as pd


def concatenate_columns(
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
