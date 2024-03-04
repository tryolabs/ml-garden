import pandas as pd

from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps import PredictStep


class XGBoostPredictStep(PredictStep):
    """Obtain the predictions for XGBoost model."""

    def execute(self, data: DataContainer) -> DataContainer:
        self.logger.debug("Obtaining predictions for XGBoost model.")

        model = data[DataContainer.MODEL]
        if model is None:
            raise Exception("Model not trained yet.")

        model_input = data[DataContainer.CLEAN]

        if not isinstance(model_input, pd.DataFrame):
            raise ValueError("model_input must be a pandas DataFrame.")

        model_configs = data.get(DataContainer.MODEL_CONFIGS)
        if model_configs:
            drop_columns: list[str] = model_configs.get("drop_columns")
            if drop_columns:
                model_input = model_input.drop(columns=drop_columns)
            target = model_configs.get("target")
            if target is None:
                raise ValueError("Target column not found in model_configs.")

            predictions = model.predict(model_input.drop(columns=[target]))
        else:
            predictions = model.predict(model_input)

        predictions_df = pd.DataFrame(predictions, columns=["prediction"])

        model_input[DataContainer.PREDICTIONS] = predictions_df

        data[DataContainer.MODEL_OUTPUT] = model_input
        return data
