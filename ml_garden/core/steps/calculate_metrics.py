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
    MAE: str
    RMSE: str
    R_2: str
    Mean_Error: str
    Max_Error: str
    Median_Absolute_Error: str


class ClassMetrics(TypedDict):
    Precision: str
    Recall: str
    F1_Score: str
    Support: str


class ClassificationOverallMetrics(TypedDict):
    Accuracy: str
    Weighted_Precision: str
    Weighted_Recall: str
    Weighted_F1_Score: str


class ClassificationMetrics(TypedDict):
    Overall: ClassificationOverallMetrics
    Per_Class: Dict[str, ClassMetrics]
    Confusion_Matrix: List[List[int]]


class DatasetMetrics(TypedDict):
    train: Dict[str, Union[RegressionMetrics, ClassificationMetrics]]
    validation: Dict[str, Union[RegressionMetrics, ClassificationMetrics]]
    test: Dict[str, Union[RegressionMetrics, ClassificationMetrics]]


class CalculateMetricsStep(PipelineStep):
    """Calculate metrics."""

    used_for_prediction = False
    used_for_training = True

    def __init__(self) -> None:
        """Initialize CalculateMetricsStep."""
        super().__init__()
        self.init_logger()

    def _calculate_regression_metrics(
        self, true_values: pd.Series, predictions: pd.Series
    ) -> RegressionMetrics:
        """Calculate regression metrics."""
        mae = mean_absolute_error(true_values, predictions)
        rmse = np.sqrt(mean_squared_error(true_values, predictions))
        r2 = r2_score(true_values, predictions)
        me = np.mean(true_values - predictions)
        max_error = np.max(np.abs(true_values - predictions))
        median_absolute_error = np.median(np.abs(true_values - predictions))

        return RegressionMetrics(
            MAE=str(mae),
            RMSE=str(rmse),
            R_2=str(r2),
            Mean_Error=str(me),
            Max_Error=str(max_error),
            Median_Absolute_Error=str(median_absolute_error),
        )

    def _calculate_classification_metrics(
        self, true_values: pd.Series, predictions: pd.Series
    ) -> ClassificationMetrics:
        """Calculate classification metrics."""
        accuracy = accuracy_score(true_values, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(true_values, predictions)
        weighted_precision = precision_score(true_values, predictions, average="weighted")
        weighted_recall = recall_score(true_values, predictions, average="weighted")
        weighted_f1 = f1_score(true_values, predictions, average="weighted")
        cm = confusion_matrix(true_values, predictions)

        overall = ClassificationOverallMetrics(
            Accuracy=str(accuracy),
            Weighted_Precision=str(weighted_precision),
            Weighted_Recall=str(weighted_recall),
            Weighted_F1_Score=str(weighted_f1),
        )

        per_class = {}
        for i, class_label in enumerate(np.unique(true_values)):
            per_class[f"Class_{class_label}"] = ClassMetrics(
                Precision=str(precision[i]),
                Recall=str(recall[i]),
                F1_Score=str(f1[i]),
                Support=str(support[i]),
            )

        return ClassificationMetrics(
            Overall=overall,
            Per_Class=per_class,
            Confusion_Matrix=cm.tolist(),
        )

    def _calculate_metrics(
        self, true_values: pd.Series, predictions: pd.Series, task: Task
    ) -> Union[RegressionMetrics, ClassificationMetrics]:
        """Calculate metrics based on the task type."""
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
        """Execute the step."""
        self.logger.debug("Starting metric calculation")

        metrics: DatasetMetrics = {}

        if data.is_train:
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

            self.logger.info(f"Metrics: {json.dumps(metrics, indent=4)}")

            data.metrics = metrics

        return data
