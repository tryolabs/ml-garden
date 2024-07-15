import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    precision_score,
    r2_score,
    recall_score,
)

from ml_garden.core import DataContainer
from ml_garden.core.constants import Task
from ml_garden.core.steps.base import PipelineStep


class CalculateMetricsStep(PipelineStep):
    """Calculate metrics."""

    used_for_prediction = False
    used_for_training = True

    def __init__(self) -> None:
        """Initialize CalculateMetricsStep."""
        super().__init__()
        self.init_logger()

    def _calculate_regression_metrics(self, true_values: pd.Series, predictions: pd.Series) -> dict:
        """Calculate regression metrics.
        Parameters
        ----------
        true_values : pd.Series
            True values
        predictions : pd.Series
            Predictions
        Returns
        -------
        dict
            Metrics
        """
        mae = mean_absolute_error(true_values, predictions)
        rmse = np.sqrt(mean_squared_error(true_values, predictions))
        r2 = r2_score(true_values, predictions)

        # Additional metrics
        me = np.mean(true_values - predictions)  # Mean Error
        max_error = np.max(np.abs(true_values - predictions))
        median_absolute_error = np.median(np.abs(true_values - predictions))

        return {
            "MAE": str(mae),
            "RMSE": str(rmse),
            "R_2": str(r2),
            "Mean Error": str(me),
            "Max Error": str(max_error),
            "Median Absolute Error": str(median_absolute_error),
        }

    def _calculate_classification_metrics(
        self, true_values: pd.Series, predictions: pd.Series
    ) -> dict:
        """Calculate classification metrics.

        Parameters
        ----------
        true_values : pd.Series
            True values
        predictions : pd.Series
            Predicted class labels

        Returns
        -------
        dict
            Classification metrics
        """
        accuracy = accuracy_score(true_values, predictions)

        # Calculate per-class precision, recall, and f1-score
        precision, recall, f1, support = precision_recall_fscore_support(true_values, predictions)

        # Overall weighted averages
        weighted_precision = precision_score(true_values, predictions, average="weighted")
        weighted_recall = recall_score(true_values, predictions, average="weighted")
        weighted_f1 = f1_score(true_values, predictions, average="weighted")

        cm = confusion_matrix(true_values, predictions)

        # Create a dictionary for per-class metrics
        class_metrics = {}
        for i, class_label in enumerate(np.unique(true_values)):
            class_metrics[f"Class_{class_label}"] = {
                "Precision": str(precision[i]),
                "Recall": str(recall[i]),
                "F1 Score": str(f1[i]),
                "Support": str(support[i]),
            }

        return {
            "Overall": {
                "Accuracy": str(accuracy),
                "Weighted Precision": str(weighted_precision),
                "Weighted Recall": str(weighted_recall),
                "Weighted F1 Score": str(weighted_f1),
            },
            "Per_Class": class_metrics,
            "Confusion Matrix": str(cm.tolist()),
        }

    def _calculate_metrics(
        self, true_values: pd.Series, predictions: pd.Series, task: Task
    ) -> dict:
        """
        Calculate metrics based on the task type.

        Parameters
        ----------
        true_values : pd.Series
            The true values of the target variable.
        predictions : pd.Series
            The predicted values of the model.
        task : Task
            The type of task, either Task.CLASSIFICATION or Task.REGRESSION.

        Returns
        -------
        dict
            A dictionary containing the calculated metrics.

        Raises
        ------
        ValueError
            If the task type is not recognized.
        """
        metric_calculators = {
            Task.CLASSIFICATION: self._calculate_classification_metrics,
            Task.REGRESSION: self._calculate_regression_metrics,
        }

        try:
            calculate_metrics = metric_calculators[task]
        except KeyError:
            raise ValueError(f"Unsupported task type: {task}")

        return calculate_metrics(true_values=true_values, predictions=predictions)

    def execute(self, data: DataContainer) -> DataContainer:
        """Execute the step.
        Parameters
        ----------
        data : DataContainer
            The data container
        Returns
        -------
        DataContainer
            The updated data container
        """
        self.logger.debug("Starting metric calculation")

        metrics = {}

        if data.is_train:
            # Metrics are only calculated during training
            for dataset_name in ["train", "validation", "test"]:
                dataset = getattr(data, dataset_name, None)

                if dataset is None:
                    self.logger.warning(
                        f"Dataset '{dataset_name}' not found. Skipping metric calculation."
                    )
                    continue

                metrics[dataset_name] = self._calculate_metrics(
                    true_values=dataset[data.target],
                    predictions=dataset[data.prediction_column],
                    task=data.task,
                )

            # pretty print metrics
            self.logger.info(f"Metrics: {json.dumps(metrics, indent=4)}")

            data.metrics = metrics

        return data
