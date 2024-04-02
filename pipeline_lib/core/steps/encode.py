from typing import Optional

import pandas as pd
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer

from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps.base import PipelineStep


class EncodeStep(PipelineStep):
    """Encode the data."""

    used_for_prediction = True
    used_for_training = True

    def __init__(self, target: Optional[str] = None, cardinality_threshold: float = 0.3) -> None:
        """Initialize EncodeStep."""
        self.init_logger()
        self.target = target
        self.cardinality_threshold = cardinality_threshold
        self.column_transformer = None

    def execute(self, data: DataContainer) -> DataContainer:
        """Execute the step."""
        self.logger.info("Encoding data")
        df = data.flow
        target_column_name = self.target or data.target

        categorical_features = [
            col for col in df.columns if df[col].dtype == "object" and col != target_column_name
        ]
        numeric_features = [
            col
            for col in df.columns
            if col not in categorical_features and col != target_column_name
        ]

        low_cardinality_features = [
            col
            for col in categorical_features
            if df[col].nunique() / len(df) < self.cardinality_threshold
        ]
        high_cardinality_features = [
            col for col in categorical_features if col not in low_cardinality_features
        ]

        self._log_feature_info(
            categorical_features,
            numeric_features,
            low_cardinality_features,
            high_cardinality_features,
        )

        self.column_transformer = ColumnTransformer(
            [
                ("target_encoder", TargetEncoder(), low_cardinality_features),
            ],
            remainder="passthrough",
            verbose_feature_names_out=True,
        )

        self.column_transformer.fit(df, df[target_column_name])
        transformed_data = self.column_transformer.transform(df)
        self.logger.info(f"Transformed data shape: {transformed_data.shape}")

        encoded_data = pd.DataFrame(transformed_data, columns=df.columns)

        data.flow = encoded_data

        return data

    def _log_feature_info(
        self,
        categorical_features,
        numeric_features,
        low_cardinality_features,
        high_cardinality_features,
    ):
        self.logger.info(f"Categorical features: {categorical_features}")
        self.logger.info(f"Numeric features: {numeric_features}")
        self.logger.info(f"Low cardinality features: {low_cardinality_features}")
        self.logger.info(f"High cardinality features: {high_cardinality_features}")
