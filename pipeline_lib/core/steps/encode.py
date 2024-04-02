from typing import Optional

import pandas as pd
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

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

        column_transformer = ColumnTransformer(
            [
                ("target_encoder", TargetEncoder(), high_cardinality_features),
                ("ordinal_encoder", OrdinalEncoder(), low_cardinality_features),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )

        column_transformer.fit(df, df[target_column_name])
        transformed_data = column_transformer.transform(df)
        self.logger.debug(f"Transformed data shape: {transformed_data.shape}")

        encoded_data = pd.DataFrame(
            transformed_data, columns=column_transformer.get_feature_names_out()
        )

        # ensure that the output columns are in the same order as the input columns
        new_column_order = [
            col for col in df.columns if col in encoded_data.columns or col == target_column_name
        ]

        encoded_data = encoded_data[new_column_order]

        data.flow = encoded_data

        return data

    def _log_feature_info(
        self,
        categorical_features,
        numeric_features,
        low_cardinality_features,
        high_cardinality_features,
    ):
        self.logger.info(
            f"Categorical features: ({len(categorical_features)}) - {categorical_features}"
        )
        self.logger.info(
            f"Low cardinality features (cardinality ratio < {self.cardinality_threshold}):"
            f" ({len(low_cardinality_features)}) - {low_cardinality_features}"
        )
        self.logger.info("Low cardinality features encoding method: ordinal encoder")
        self.logger.info(
            f"High cardinality features (cardinality ratio >= {self.cardinality_threshold}):"
            f" ({len(high_cardinality_features)}) -  {high_cardinality_features}"
        )
        self.logger.info("High cardinality features encoding method: target encoder")
        self.logger.info(f"Numeric features: ({len(numeric_features)}) - {numeric_features}")
        self.logger.info("Numeric features encoding method: passthrough")
