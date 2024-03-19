from typing import Optional

from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps.base import PipelineStep


class CleanStep(PipelineStep):
    def __init__(
        self,
        fill_missing: Optional[dict] = None,
        remove_outliers: Optional[dict] = None,
        convert_dtypes: Optional[dict] = None,
        drop_na_columns: Optional[list] = None,
        drop_ids: Optional[dict] = None,
    ):
        self.init_logger()
        self.fill_missing = fill_missing
        self.remove_outliers = remove_outliers
        self.convert_dtypes = convert_dtypes
        self.drop_na_columns = drop_na_columns
        self.drop_ids = drop_ids

    def execute(self, data: DataContainer) -> DataContainer:
        self.logger.info("Cleaning tabular data...")

        df = data.raw

        if self.fill_missing:
            for column, fill_value in self.fill_missing.items():
                if column in df.columns:
                    df[column].fillna(fill_value, inplace=True)
                    self.logger.info(
                        f"Filled missing values in column '{column}' with {fill_value}"
                    )
                else:
                    self.logger.warning(f"Column '{column}' not found in the DataFrame")

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
                        self.logger.info(f"Clipped outliers in column '{column}'")
                    elif method == "drop":
                        q1 = df[column].quantile(0.25)
                        q3 = df[column].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - (1.5 * iqr)
                        upper_bound = q3 + (1.5 * iqr)
                        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
                        df = df[~outliers]
                        self.logger.info(f"Dropped outliers in column '{column}'")
                    else:
                        self.logger.warning(f"Unsupported outlier removal method '{method}'")
                else:
                    self.logger.warning(f"Column '{column}' not found in the DataFrame")

        if self.convert_dtypes:
            for column, dtype in self.convert_dtypes.items():
                if column in df.columns:
                    df[column] = df[column].astype(dtype)
                    self.logger.info(f"Converted column '{column}' to {dtype}")
                else:
                    self.logger.warning(f"Column '{column}' not found in the DataFrame")

        if self.drop_na_columns:
            for column in self.drop_na_columns:
                if column in df.columns:
                    initial_rows = len(df)
                    df.dropna(subset=[column], inplace=True)
                    dropped_rows = initial_rows - len(df)
                    self.logger.info(
                        f"Dropped {dropped_rows} rows with None values in column '{column}'"
                    )
                else:
                    self.logger.warning(f"Column '{column}' not found in the DataFrame")

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
                            f"Dropped {dropped_rows} rows ({percentage_dropped:.2f}%) with IDs"
                            f" {list(dropped_ids)} in column '{column}'"
                        )
                    else:
                        self.logger.info(
                            f"No rows dropped for IDs {list(ids)} in column '{column}'"
                        )

                    if not_found_ids:
                        self.logger.warning(
                            f"IDs {list(not_found_ids)} not found in column '{column}'"
                        )
                else:
                    self.logger.warning(f"Column '{column}' not found in the DataFrame")

        data.clean = df
        data.flow = df

        return data
