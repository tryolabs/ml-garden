import pandas as pd
from joblib import load

from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps import PredictStep


class XGBoostPredictStep(PredictStep):
    """Obtain the predictions for XGBoost model."""

    def execute(self, data: DataContainer) -> DataContainer:
        self.logger.debug("Obtaining predictions for XGBoost model.")

        if not self.config:
            raise ValueError("No prediction configs found.")

        load_path = self.config.get("load_path")
        if not load_path:
            raise ValueError("No load path found in model_configs.")

        if not load_path.endswith(".joblib"):
            raise ValueError("Only joblib format is supported for loading the model.")
        model = load(load_path)

        model_input = data[DataContainer.CLEAN]

        if not isinstance(model_input, pd.DataFrame):
            raise ValueError("model_input must be a pandas DataFrame.")

        if self.config:
            drop_columns = self.config.get("drop_columns")
            if drop_columns:
                model_input = model_input.drop(columns=drop_columns)

            target = self.config.get("target")
            if target is None:
                raise ValueError("Target column not found in model_configs.")
            data[DataContainer.TARGET] = target

            predictions = model.predict(model_input.drop(columns=[target]))
        else:
            predictions = model.predict(model_input)

        predictions_df = pd.DataFrame(predictions, columns=["prediction"])

        model_input[DataContainer.PREDICTIONS] = predictions_df

        data[DataContainer.MODEL_OUTPUT] = model_input
        return data
