from typing import Optional

import pandas as pd
from joblib import load

from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps import PredictStep


class XGBoostPredictStep(PredictStep):
    """Obtain the predictions for XGBoost model."""

    def __init__(
        self,
        target: str,
        load_path: str,
        drop_columns: Optional[list[str]] = None,
    ) -> None:
        self.init_logger()

        if not load_path.endswith(".joblib"):
            raise ValueError("Only joblib format is supported for loading the model.")

        self.target = target
        self.load_path = load_path
        self.drop_columns = drop_columns

        self.model = load(self.load_path)

    def execute(self, data: DataContainer) -> DataContainer:
        self.logger.debug("Obtaining predictions for XGBoost model.")

        model_input = data[DataContainer.CLEAN]

        if self.drop_columns:
            model_input = model_input.drop(columns=self.drop_columns)

        predictions = self.model.predict(model_input.drop(columns=[self.target]))

        predictions_df = pd.DataFrame(predictions, columns=["prediction"])

        model_input[DataContainer.PREDICTIONS] = predictions_df
        data[DataContainer.MODEL] = self.model
        data[DataContainer.MODEL_OUTPUT] = model_input
        data[DataContainer.TARGET] = self.target
        data[DataContainer.DROP_COLUMNS] = self.drop_columns
        return data
