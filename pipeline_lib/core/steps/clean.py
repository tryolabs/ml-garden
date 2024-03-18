from typing import Optional

from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps.base import PipelineStep


class CleanStep(PipelineStep):
    def __init__(
        self,
        fill_missing: Optional[dict] = None,
        remove_outliers: Optional[dict] = None,
        convert_dtypes: Optional[dict] = None,
    ):
        self.init_logger()
        self.fill_missing = fill_missing
        self.remove_outliers = remove_outliers
        self.convert_dtypes = convert_dtypes

    def execute(self, data: DataContainer) -> DataContainer:
        self.logger.info("Cleaning tabular data...")

        df = data[DataContainer.RAW]

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

        data[DataContainer.CLEAN] = df

        return data
