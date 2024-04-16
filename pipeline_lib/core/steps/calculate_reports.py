import time

import numpy as np
import pandas as pd
import shap
from tqdm import tqdm

from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps.base import PipelineStep


class CalculateReportsStep(PipelineStep):
    """Calculate reports."""

    used_for_prediction = True
    used_for_training = False

    def __init__(self, max_samples: int = 1000) -> None:
        """Initialize CalculateReportsStep."""
        self.init_logger()
        self.max_samples = max_samples

    def execute(self, data: DataContainer) -> DataContainer:
        """Execute the step."""
        self.logger.info("Calculating reports")

        model = data.model
        if model is None:
            raise ValueError("Model not found in data container.")

        df = data.flow
        if len(df) > self.max_samples:
            # Randomly sample a subset of data points if the dataset is larger than max_samples
            self.logger.info(
                f"Dataset contains {len(df)} data points and max_samples is set to"
                f" {self.max_samples}."
            )
            self.logger.info(f"Sampling {self.max_samples} data points from the dataset.")
            df = df.sample(n=self.max_samples, random_state=42)

        drop_columns = (
            data._drop_columns + ["predictions"] if data._drop_columns else ["predictions"]
        )
        df = df.drop(columns=drop_columns)
        X = df.drop(columns=[data.target])

        # Calculate SHAP values with progress tracking and logging
        explainer = shap.TreeExplainer(model.model)
        shap_values = []
        # shap_base_value = explainer.expected_value
        total_rows = len(X)
        start_time = time.time()
        with tqdm(total=total_rows, desc="Calculating SHAP values") as pbar:
            for i in range(total_rows):
                shap_value = explainer.shap_values(X.iloc[[i]])
                shap_values.append(shap_value[0])  # Append only the first element of shap_value
                pbar.update(1)
                elapsed_time = time.time() - start_time
                remaining_time = (elapsed_time / (i + 1)) * (total_rows - i - 1)
                pbar.set_postfix(elapsed=f"{elapsed_time:.2f}s", remaining=f"{remaining_time:.2f}s")

        shap_values = np.array(shap_values)  # Convert shap_values to a NumPy array

        feature_names = X.columns.tolist()
        feature_importance = pd.DataFrame(
            list(zip(feature_names, abs(shap_values).mean(0))),
            columns=["feature", "importance"],
        )
        feature_importance = feature_importance.sort_values(by="importance", ascending=True)
        feature_importance.reset_index(drop=True, inplace=True)

        data.feature_importance = feature_importance

        return data
