from enum import Enum


class Task(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    QUANTILE_REGRESSION = "quantile"
