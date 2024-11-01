"""Calculate metrics for regression and classification tasks."""

import json
from typing import Dict, List, TypedDict, Union

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


class RegressionMetrics(TypedDict):
    """Regression metrics."""

    MAE: float
    RMSE: float
    R_2: float
    Mean_Error: float
    Max_Error: float
    Median_Absolute_Error: float


class ClassMetrics(TypedDict):
    """Classification metrics."""

    Precision: float
    Recall: float
    F1_Score: float
    Support: int


class ClassificationOverallMetrics(TypedDict):
    """Classification overall metrics."""

    Accuracy: float
    Weighted_Precision: float
    Weighted_Recall: float
    Weighted_F1_Score: float


class ClassificationMetrics(TypedDict):
    """Classification metrics."""

    Overall: ClassificationOverallMetrics
    Per_Class: Dict[str, ClassMetrics]
    Confusion_Matrix: List[List[int]]


class DatasetMetrics(TypedDict):
    """Dataset metrics."""

    train: Dict[str, Union[RegressionMetrics, ClassificationMetrics]]
    validation: Dict[str, Union[RegressionMetrics, ClassificationMetrics]]
    test: Dict[str, Union[RegressionMetrics, ClassificationMetrics]]


class CalculateMetricsStep(PipelineStep):
    """Calculate metrics for regression and classification tasks."""

    used_for_prediction = False
    used_for_training = True

    def __init__(self) -> None:
        """Initialize CalculateMetricsStep."""
        super().__init__()
        self.init_logger()

    def _calculate_regression_metrics(
        self, true_values: pd.Series, predictions: pd.Series
    ) -> RegressionMetrics:
        """
        Calculate regression metrics.

        Parameters
        ----------
        true_values : pd.Series
            The true values of the target variable.
        predictions : pd.Series
            The predicted values of the target variable.

        Returns
        -------
        RegressionMetrics
            A dictionary containing various regression metrics.
        """
        mae = mean_absolute_error(true_values, predictions)
        rmse = np.sqrt(mean_squared_error(true_values, predictions))
        r2 = r2_score(true_values, predictions)
        me = np.mean(true_values - predictions)
        max_error = np.max(np.abs(true_values - predictions))
        median_absolute_error = np.median(np.abs(true_values - predictions))

        return RegressionMetrics(
            MAE=float(mae),
            RMSE=float(rmse),
            R_2=float(r2),
            Mean_Error=float(me),
            Max_Error=float(max_error),
            Median_Absolute_Error=float(median_absolute_error),
        )

    def _calculate_classification_metrics(
        self, true_values: pd.Series, predictions: pd.Series
    ) -> ClassificationMetrics:
        """
        Calculate classification metrics.

        Parameters
        ----------
        true_values : pd.Series
            The true values of the target variable.
        predictions : pd.Series
            The predicted values of the target variable.

        Returns
        -------
        ClassificationMetrics
            A dictionary containing various classification metrics.
        """
        accuracy = accuracy_score(true_values, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(true_values, predictions)
        weighted_precision = precision_score(true_values, predictions, average="weighted")
        weighted_recall = recall_score(true_values, predictions, average="weighted")
        weighted_f1 = f1_score(true_values, predictions, average="weighted")
        cm = confusion_matrix(true_values, predictions)

        overall = ClassificationOverallMetrics(
            Accuracy=float(accuracy),
            Weighted_Precision=float(weighted_precision),
            Weighted_Recall=float(weighted_recall),
            Weighted_F1_Score=float(weighted_f1),
        )

        per_class = {}
        for i, class_label in enumerate(np.unique(true_values)):
            per_class[str(class_label)] = ClassMetrics(
                Precision=float(precision[i]),
                Recall=float(recall[i]),
                F1_Score=float(f1[i]),
                Support=int(support[i]),
            )

        return ClassificationMetrics(
            Overall=overall,
            Per_Class=per_class,
            Confusion_Matrix=cm.tolist(),
        )

    def _calculate_metrics(
        self, true_values: pd.Series, predictions: pd.Series, task: Task
    ) -> Union[RegressionMetrics, ClassificationMetrics]:
        """
        Calculate metrics based on the task type.

        Parameters
        ----------
        true_values : pd.Series
            The true values of the target variable.
        predictions : pd.Series
            The predicted values of the target variable.
        task : Task
            The type of machine learning task (classification or regression).

        Returns
        -------
        Union[RegressionMetrics, ClassificationMetrics]
            A dictionary containing the calculated metrics.

        Raises
        ------
        ValueError
            If an unsupported task type is provided.
        """
        metric_calculators = {
            Task.CLASSIFICATION: self._calculate_classification_metrics,
            Task.REGRESSION: self._calculate_regression_metrics,
        }

        try:
            calculate_metrics = metric_calculators[task]
        except KeyError:
            error_message = f"Unsupported task type: {task}"
            raise ValueError(error_message) from None

        return calculate_metrics(true_values=true_values, predictions=predictions)

    def execute(self, data: DataContainer) -> DataContainer:
        """
        Execute the metric calculation step.

        Parameters
        ----------
        data : DataContainer
            The data container object containing the datasets and metadata.

        Returns
        -------
        DataContainer
            The updated data container with calculated metrics.
        """
        self.logger.debug("Starting metric calculation")

        metrics: Dict[str, Union[RegressionMetrics, ClassificationMetrics]] = {}

        if data.is_train:
            for dataset_name in ["train", "validation", "test"]:
                dataset = getattr(data, dataset_name, None)

                if dataset is None:
                    self.logger.warning(
                        "Dataset '%s' not found. Skipping metric calculation.", dataset_name
                    )
                    continue

                metrics[dataset_name] = self._calculate_metrics(
                    true_values=dataset[data.target],
                    predictions=dataset[data.prediction_column],
                    task=data.task,
                )

            self.logger.info("Metrics: %s", json.dumps(metrics, indent=4))

            data.metrics = metrics

        return data
