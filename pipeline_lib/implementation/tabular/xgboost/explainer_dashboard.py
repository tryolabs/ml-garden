from explainerdashboard import RegressionExplainer

from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps import ExplainerDashboardStep


class XGBoostExplainerDashboardStep(ExplainerDashboardStep):
    """Scale the target using Quantile Transformer."""

    def execute(self, data: DataContainer) -> DataContainer:
        self.logger.debug("Starting explainer dashboard")

        model = data.get(DataContainer.MODEL)
        if model is None:
            raise ValueError("Model not found in data container.")

        val_df = data.get(DataContainer.VALIDATION)
        if val_df is None:
            raise ValueError("Validation data not found in data container.")

        model_configs = data[DataContainer.MODEL_CONFIGS]
        if model_configs is None:
            raise ValueError("Model configs not found in data container.")

        target = model_configs.get("target")
        if target is None:
            raise ValueError("Target column not found in model_configs.")

        drop_columns = model_configs.get("drop_columns")
        if drop_columns:
            val_df = val_df.drop(columns=drop_columns)

        X_test = val_df.drop(columns=[target])
        y_test = val_df[target]

        explainer = RegressionExplainer(model, X_test, y_test)

        data[DataContainer.EXPLAINER] = explainer

        return data
