import json
from typing import List, Optional, Tuple, Union

import numpy as np
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

    def __init__(
        self,
        target: Optional[str] = None,
        cardinality_threshold: float = 0.3,
        low_cardinality_encoder: str = "OrdinalEncoder",
        high_cardinality_encoder: str = "TargetEncoder",
    ) -> None:
        """Initialize EncodeStep."""
        self.init_logger()
        self.target = target
        self.cardinality_threshold = cardinality_threshold
        self.low_cardinality_encoder = low_cardinality_encoder
        self.high_cardinality_encoder = high_cardinality_encoder
        self.encoder_feature_map = {}

    def execute(self, data: DataContainer) -> DataContainer:
        """Execute the encoding step."""
        self.logger.info("Encoding data")
        df = data.flow
        target_column_name = self.target or data.target
        target_original_dtype = None

        categorical_features, numeric_features = self._get_feature_types(df, target_column_name)
        low_cardinality_features, high_cardinality_features = self._split_categorical_features(
            df, categorical_features
        )

        original_numeric_dtypes = {col: df[col].dtype for col in numeric_features}

        if pd.api.types.is_numeric_dtype(df[target_column_name]):
            target_original_dtype = df[target_column_name].dtype

        column_transformer = self._create_column_transformer(
            high_cardinality_features, low_cardinality_features
        )

        encoded_data = self._transform_data(df, target_column_name, column_transformer)
        encoded_data = self._restore_column_order(df, encoded_data)
        encoded_data = self._convert_ordinal_encoded_columns_to_int(encoded_data)
        encoded_data = self._restore_numeric_dtypes(encoded_data, original_numeric_dtypes)
        encoded_data = self._restore_target_dtype(
            encoded_data, target_column_name, target_original_dtype
        )
        encoded_data = self._convert_float64_to_float32(encoded_data)

        self._log_feature_info(
            categorical_features,
            numeric_features,
            low_cardinality_features,
            high_cardinality_features,
        )

        data.flow = encoded_data

        return data

    def _get_feature_types(
        self, df: pd.DataFrame, target_column_name: str
    ) -> Tuple[List[str], List[str]]:
        """Get categorical and numeric feature lists."""
        categorical_features = [
            col for col in df.columns if df[col].dtype == "object" and col != target_column_name
        ]
        numeric_features = [
            col
            for col in df.columns
            if col not in categorical_features and col != target_column_name
        ]
        return categorical_features, numeric_features

    def _split_categorical_features(
        self, df: pd.DataFrame, categorical_features: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Split categorical features into low and high cardinality features."""
        low_cardinality_features = [
            col
            for col in categorical_features
            if df[col].nunique() / len(df) < self.cardinality_threshold
        ]
        high_cardinality_features = [
            col for col in categorical_features if col not in low_cardinality_features
        ]
        return low_cardinality_features, high_cardinality_features

    def _get_encoder(self, encoder_name: str) -> Union[OrdinalEncoder, TargetEncoder]:
        """Map encoder name to the corresponding encoder class."""
        encoder_map = {
            "OrdinalEncoder": OrdinalEncoder(),
            "TargetEncoder": TargetEncoder(),
            # Add more encoders as needed
        }

        encoder = encoder_map.get(encoder_name)

        if not encoder:
            raise ValueError(
                f"Unsupported encoder: {encoder_name}. Supported encoders:"
                f" {list(encoder_map.keys())}"
            )

        return encoder

    def _create_column_transformer(
        self, high_cardinality_features: List[str], low_cardinality_features: List[str]
    ) -> ColumnTransformer:
        """Create a ColumnTransformer for encoding."""
        high_cardinality_encoder = self._get_encoder(self.high_cardinality_encoder)
        low_cardinality_encoder = self._get_encoder(self.low_cardinality_encoder)

        # Initialize the encoder_feature_map as an empty dictionary
        self.encoder_feature_map = {}

        # Check if both encoders are the same
        if self.high_cardinality_encoder == self.low_cardinality_encoder:
            # If the same, merge the feature lists
            # This assumes you want to combine the features into a single list; adjust if needed
            combined_features = high_cardinality_features + low_cardinality_features
            self.encoder_feature_map[self.high_cardinality_encoder] = combined_features
        else:
            # If not the same, assign individually
            self.encoder_feature_map[self.high_cardinality_encoder] = high_cardinality_features
            self.encoder_feature_map[self.low_cardinality_encoder] = low_cardinality_features

        self.logger.info(f"Encoder feature map: \n{json.dumps(self.encoder_feature_map, indent=4)}")

        return ColumnTransformer(
            [
                ("high_cardinality_encoder", high_cardinality_encoder, high_cardinality_features),
                ("low_cardinality_encoder", low_cardinality_encoder, low_cardinality_features),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )

    def _transform_data(
        self, df: pd.DataFrame, target_column_name: str, column_transformer: ColumnTransformer
    ) -> pd.DataFrame:
        """Transform the data using the ColumnTransformer."""
        column_transformer.fit(df, df[target_column_name])
        transformed_data = column_transformer.transform(df)
        self.logger.debug(f"Transformed data shape: {transformed_data.shape}")
        return pd.DataFrame(transformed_data, columns=column_transformer.get_feature_names_out())

    def _restore_column_order(self, df: pd.DataFrame, encoded_data: pd.DataFrame) -> pd.DataFrame:
        """Restore the original column order."""
        new_column_order = [col for col in df.columns if col in encoded_data.columns]
        return encoded_data[new_column_order]

    def _convert_ordinal_encoded_columns_to_int(self, encoded_data: pd.DataFrame) -> pd.DataFrame:
        """Convert ordinal encoded columns to the smallest possible integer dtype."""
        ordinal_encoded_features = self.encoder_feature_map.get("OrdinalEncoder", [])

        for col in ordinal_encoded_features:
            if col in encoded_data.columns:
                n_unique = encoded_data[col].nunique()
                if n_unique <= 2**8:
                    encoded_data[col] = encoded_data[col].astype(np.int8)
                elif n_unique <= 2**16:
                    encoded_data[col] = encoded_data[col].astype(np.int16)
                elif n_unique <= 2**32:
                    encoded_data[col] = encoded_data[col].astype(np.int32)
                else:
                    encoded_data[col] = encoded_data[col].astype(np.int64)

        return encoded_data

    def _restore_numeric_dtypes(
        self, encoded_data: pd.DataFrame, original_numeric_dtypes: dict
    ) -> pd.DataFrame:
        """Restore original dtypes of numeric features."""
        for col, dtype in original_numeric_dtypes.items():
            if col in encoded_data.columns:
                try:
                    encoded_data[col] = encoded_data[col].astype(dtype)
                except ValueError:
                    self.logger.warning(
                        f"Failed to convert column '{col}' to its original dtype ({dtype})."
                    )
        return encoded_data

    def _restore_target_dtype(
        self,
        encoded_data: pd.DataFrame,
        target_column_name: str,
        target_original_dtype: Optional[np.dtype],
    ) -> pd.DataFrame:
        """Restore original dtype of the target column."""
        if not target_original_dtype:
            return encoded_data

        try:
            encoded_data[target_column_name] = encoded_data[target_column_name].astype(
                target_original_dtype
            )
        except ValueError:
            self.logger.warning(
                f"Failed to convert target column '{target_column_name}' to its original dtype"
                f" ({target_original_dtype})."
            )
        return encoded_data

    def _convert_float64_to_float32(self, encoded_data: pd.DataFrame) -> pd.DataFrame:
        """Convert float64 columns to float32."""
        float64_columns = encoded_data.select_dtypes(include=["float64"]).columns
        for col in float64_columns:
            encoded_data[col] = encoded_data[col].astype(np.float32)
        return encoded_data

    def _log_feature_info(
        self,
        categorical_features: List[str],
        numeric_features: List[str],
        low_cardinality_features: List[str],
        high_cardinality_features: List[str],
    ) -> None:
        """Log information about the features."""
        self.logger.info(
            f"Categorical features: ({len(categorical_features)}) - {categorical_features}"
        )
        self.logger.info(
            f"Low cardinality features (cardinality ratio < {self.cardinality_threshold}):"
            f" ({len(low_cardinality_features)}) - {low_cardinality_features}"
        )
        self.logger.info(
            f"High cardinality features (cardinality ratio >= {self.cardinality_threshold}):"
            f" ({len(high_cardinality_features)}) -  {high_cardinality_features}"
        )
        self.logger.info(f"Numeric features: ({len(numeric_features)}) - {numeric_features}")
