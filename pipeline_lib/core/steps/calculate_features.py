from typing import List, Optional

from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps.base import PipelineStep


class CalculateFeaturesStep(PipelineStep):
    """Calculate features."""

    def __init__(
        self,
        datetime_columns: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        config: Optional[dict] = None,
    ) -> None:
        """Initialize CalculateFeaturesStep."""
        super().__init__(config=config)
        self.init_logger()
        self.datetime_columns = datetime_columns
        self.features = features

    def execute(self, data: DataContainer) -> DataContainer:
        """Execute the step."""
        self.logger.info("Calculating features")

        df = data[DataContainer.CLEAN]

        if self.datetime_columns:
            for column in self.datetime_columns:
                if column in df.columns:
                    if self.features:
                        for feature in self.features:
                            if feature == "year":
                                df.loc[:, f"{column}_year"] = df[column].dt.year
                            elif feature == "month":
                                df.loc[:, f"{column}_month"] = df[column].dt.month
                            elif feature == "day":
                                df.loc[:, f"{column}_day"] = df[column].dt.day
                            elif feature == "hour":
                                df.loc[:, f"{column}_hour"] = df[column].dt.hour
                            elif feature == "minute":
                                df.loc[:, f"{column}_minute"] = df[column].dt.minute
                            elif feature == "second":
                                df.loc[:, f"{column}_second"] = df[column].dt.second
                            elif feature == "weekday":
                                df.loc[:, f"{column}_weekday"] = df[column].dt.weekday
                            elif feature == "dayofyear":
                                df.loc[:, f"{column}_dayofyear"] = df[column].dt.dayofyear
                            else:
                                self.logger.warning(f"Unsupported datetime feature: {feature}")
                    else:
                        self.logger.warning(
                            "No datetime features specified. Skipping feature extraction."
                        )
                else:
                    self.logger.warning(f"Datetime column '{column}' not found in the DataFrame")
        else:
            self.logger.warning("No datetime columns specified. Skipping feature extraction.")

        # drop original datetime columns
        if self.datetime_columns:
            df = df.drop(columns=self.datetime_columns)
            self.logger.info(f"Dropped datetime columns: {self.datetime_columns}")

        data[DataContainer.FEATURES] = df

        return data
