import json
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

from ml_garden.core import DataContainer
from ml_garden.core.steps.base import PipelineStep


class EncodeStep(PipelineStep):
    """Encode the data."""

    used_for_prediction = True
    used_for_training = True

    HIGH_CARDINALITY_ENCODER = "TargetEncoder"
    LOW_CARDINALITY_ENCODER = "OrdinalEncoder"

    ENCODER_MAP = {
        "OrdinalEncoder": OrdinalEncoder,
        "TargetEncoder": TargetEncoder,
    }
    ENCODER_MAP_PARAMS = {
        # Default to -1 for unknown categories in OrdinalEncoders
        "OrdinalEncoder": {"handle_unknown": "use_encoded_value", "unknown_value": -1},
        "TargetEncoder": {},
    }

    def __init__(
        self,
        cardinality_threshold: int = 5,
        feature_encoders: Optional[dict] = None,
    ) -> None:
        """Initialize EncodeStep.
        Parameters
        ----------
        cardinality_threshold : int, optional
            The threshold to determine low and high cardinality features, by default 5
        feature_encoders : Optional[dict], optional
            A dictionary mapping feature names to encoder configurations, by default None
        """
        self.init_logger()
        self.cardinality_threshold = cardinality_threshold
        self.feature_encoders = feature_encoders or {}

    def execute(self, data: DataContainer) -> DataContainer:
        """Execute the encoding step."""
        self.logger.info("Encoding data")

        target_column_name = data.target

        if not data.is_train:
            categorical_features, numeric_features = self._get_feature_types(
                data.flow.drop(columns=data.columns_to_ignore_for_training), data.target
            )
            data.X_prediction, _, _ = self._apply_encoding(
                data.flow,
                target_column_name,
                data.columns_to_ignore_for_training,
                categorical_features,
                numeric_features,
                saved_encoder=data._encoder,
                log=True,
            )
            return data

        categorical_features, numeric_features = self._get_feature_types(
            data.train.drop(columns=data.columns_to_ignore_for_training), target_column_name
        )

        data.X_train, data.y_train, data._encoder = self._apply_encoding(
            data.train,
            target_column_name,
            data.columns_to_ignore_for_training,
            categorical_features,
            numeric_features,
            fit_encoders=True,
            log=True,
        )

        if data.validation is not None:
            data.X_validation, data.y_validation, _ = self._apply_encoding(
                data.validation,
                target_column_name,
                data.columns_to_ignore_for_training,
                categorical_features,
                numeric_features,
                saved_encoder=data._encoder,
            )

        if data.test is not None:
            data.X_test, data.y_test, _ = self._apply_encoding(
                data.test,
                target_column_name,
                data.columns_to_ignore_for_training,
                categorical_features,
                numeric_features,
                saved_encoder=data._encoder,
            )

        return data

    def _apply_encoding(
        self,
        df: pd.DataFrame,
        target_column_name: str,
        columns_to_ignore_for_training: List[str],
        categorical_features: List[str],
        numeric_features: List[str],
        fit_encoders: bool = False,
        saved_encoder: Optional[ColumnTransformer] = None,
        log: Optional[bool] = False,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[ColumnTransformer]]:
        """Apply the encoding to the data.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to encode
        target_column_name : str
            The target column name
        columns_to_ignore_for_training : List[str]
            Columns to ignore for training
        categorical_features : List[str]
            Categorical features
        numeric_features : List[str]
            Numeric features
        fit_encoders : bool, optional
            Whether to fit the encoders, by default False
        saved_encoder : Optional[ColumnTransformer], optional
            The saved encoder, by default None
        log : Optional[bool], optional
            Whether to log information about the features, by default False

        Returns
        -------
        Tuple[pd.DataFrame, Optional[pd.Series], Optional[ColumnTransformer]]
            The encoded data, the target column, and the encoder
        """
        if not fit_encoders and not saved_encoder:
            raise ValueError("saved_encoder must be provided when fit_encoders is False.")

        df = df.drop(columns=columns_to_ignore_for_training)

        low_cardinality_features, high_cardinality_features = self._split_categorical_features(
            df, categorical_features
        )
        original_numeric_dtypes = {col: df[col].dtype for col in numeric_features}

        if fit_encoders:
            # Save the encoder for prediction
            encoder = self._create_column_transformer(
                high_cardinality_features,
                low_cardinality_features,
                numeric_features,
            )
        else:
            encoder = saved_encoder
            assert encoder is not None

        encoded_data, targets = self._transform_data(
            df,
            target_column_name,
            encoder,
            fit_encoders,
        )

        encoded_data = self._restore_column_order(df, encoded_data)
        encoded_data = self._restore_numeric_dtypes(encoded_data, original_numeric_dtypes)
        encoded_data = self._convert_float64_to_float32(encoded_data)

        feature_encoder_map = self._create_feature_encoder_map(encoder)
        encoded_data = self._convert_ordinal_encoded_columns_to_int(
            encoded_data, feature_encoder_map
        )

        if log:
            self._log_feature_info(
                categorical_features,
                numeric_features,
                low_cardinality_features,
                high_cardinality_features,
                feature_encoder_map,
            )

        return encoded_data, targets, encoder

    def _get_feature_types(
        self, df: pd.DataFrame, target_column_name: str
    ) -> Tuple[List[str], List[str]]:
        """Get categorical and numeric feature lists."""
        categorical_features = [
            col
            for col in df.columns
            if df[col].dtype in ["object", "category"] and col != target_column_name
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
            col for col in categorical_features if df[col].nunique() <= self.cardinality_threshold
        ]
        high_cardinality_features = [
            col for col in categorical_features if col not in low_cardinality_features
        ]
        return low_cardinality_features, high_cardinality_features

    def _get_encoder_class_and_params(
        self, encoder_name: str
    ) -> Tuple[Union[Type[OrdinalEncoder], Type[TargetEncoder]], dict[str, Any]]:
        """Map encoder name to the corresponding encoder class."""
        encoder = self.ENCODER_MAP.get(encoder_name)
        encoder_params = self.ENCODER_MAP_PARAMS.get(encoder_name)

        if not encoder or encoder_params is None:
            raise ValueError(
                f"Unsupported encoder: {encoder_name}. Supported encoders:"
                f" {list(self.ENCODER_MAP.keys())}"
            )

        return encoder, encoder_params

    def _log_encoder_override(
        self,
        feature: str,
        encoder_class: Type[Union[OrdinalEncoder, TargetEncoder]],
        high_cardinality_features: List[str],
        low_cardinality_features: List[str],
    ):
        if feature in high_cardinality_features:
            self.logger.info(
                f"Feature '{feature}' encoder overridden from"
                f" {self.HIGH_CARDINALITY_ENCODER} to {encoder_class.__name__}"
            )
        elif feature in low_cardinality_features:
            self.logger.info(
                f"Feature '{feature}' encoder overridden from {self.LOW_CARDINALITY_ENCODER} to"
                f" {encoder_class.__name__}"
            )
        else:
            self.logger.info(
                f"Feature '{feature}' explicitly encoded with {encoder_class.__name__}"
            )

    def _create_column_transformer(
        self,
        high_cardinality_features: List[str],
        low_cardinality_features: List[str],
        numeric_features: List[str],
    ) -> ColumnTransformer:
        """Create a ColumnTransformer for encoding."""
        transformers = []

        for feature in high_cardinality_features + low_cardinality_features:
            if feature in self.feature_encoders:
                encoder_config = self.feature_encoders[feature]
                encoder_class, encoder_params = self._get_encoder_class_and_params(
                    encoder_config["encoder"]
                )
                encoder_params.update(encoder_config.get("params", {}))
                self._log_encoder_override(
                    feature, encoder_class, high_cardinality_features, low_cardinality_features
                )
            elif feature in high_cardinality_features:
                encoder_class, encoder_params = self._get_encoder_class_and_params(
                    self.HIGH_CARDINALITY_ENCODER
                )
            else:
                encoder_class, encoder_params = self._get_encoder_class_and_params(
                    self.LOW_CARDINALITY_ENCODER
                )

            encoder = encoder_class(**encoder_params)
            transformers.append((f"{feature}_encoder", encoder, [feature]))

        if numeric_features:
            transformers.append(("numeric", "passthrough", numeric_features))

        return ColumnTransformer(
            transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )

    def _transform_data(
        self,
        df: pd.DataFrame,
        target_column_name: str,
        column_transformer: ColumnTransformer,
        is_train: bool = False,
    ) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        """Transform the data using the ColumnTransformer."""
        if target_column_name in df.columns:
            X = df.drop(columns=[target_column_name])  # Drop the target column
            y = df[target_column_name]  # Target column for training data
        else:
            X = df  # All columns for prediction data
            y = None  # No target in the prediction data

        if is_train:
            self.logger.info("Fitting encoders")
            column_transformer.fit(X, y)

        transformed_data = column_transformer.transform(X)
        self.logger.debug(f"Transformed data shape: {transformed_data.shape}")
        return (
            pd.DataFrame(
                transformed_data, columns=column_transformer.get_feature_names_out(), index=df.index
            ),
            y,
        )

    def _restore_column_order(self, df: pd.DataFrame, encoded_data: pd.DataFrame) -> pd.DataFrame:
        """Restore the original column order."""
        new_column_order = [col for col in df.columns if col in encoded_data.columns]
        return encoded_data[new_column_order]

    def _convert_ordinal_encoded_columns_to_int(
        self, encoded_data: pd.DataFrame, encoded_feature_map: Dict[str, str]
    ) -> pd.DataFrame:
        """Convert ordinal encoded columns to the smallest possible integer dtype."""
        ordinal_encoded_features = [
            col for col, encoder in encoded_feature_map.items() if encoder == "OrdinalEncoder"
        ]
        for col in ordinal_encoded_features:
            if col in encoded_data.columns:
                try:
                    encoded_data[col] = pd.to_numeric(encoded_data[col].values, downcast="unsigned")
                except ValueError:
                    try:
                        encoded_data[col] = pd.to_numeric(
                            encoded_data[col].values, downcast="integer"
                        )
                    except ValueError:
                        try:
                            encoded_data[col] = pd.to_numeric(
                                encoded_data[col].values, downcast="float"
                            )
                        except ValueError:
                            encoded_data[col] = encoded_data[col].astype(pd.StringDtype())

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

    def _convert_float64_to_float32(self, encoded_data: pd.DataFrame) -> pd.DataFrame:
        """Convert float64 columns to float32."""
        float64_columns = encoded_data.select_dtypes(include=["float64"]).columns
        for col in float64_columns:
            encoded_data[col] = encoded_data[col].astype(np.float32)
        return encoded_data

    def _create_feature_encoder_map(self, column_transformer: ColumnTransformer) -> Dict[str, str]:
        """Create a dictionary to store the encoder used for each feature."""
        feature_encoder_map = {}
        transformed_features = column_transformer.get_feature_names_out()

        for transformer_name, transformer, features in column_transformer.transformers_:
            encoder_name = (
                "PassThrough" if transformer_name == "numeric" else transformer.__class__.__name__
            )

            for feature in features:
                if feature in transformed_features:
                    feature_encoder_map[feature] = encoder_name

        return feature_encoder_map

    def _log_feature_info(
        self,
        categorical_features: List[str],
        numeric_features: List[str],
        low_cardinality_features: List[str],
        high_cardinality_features: List[str],
        feature_encoder_map: Dict[str, str],
    ) -> None:
        """Log information about the features."""
        self.logger.info(
            f"Categorical features: ({len(categorical_features)}) - {categorical_features}"
        )
        self.logger.info(
            f"Low cardinality features (#unique classes <= {self.cardinality_threshold}):"
            f" ({len(low_cardinality_features)}) - {low_cardinality_features}"
        )
        self.logger.info(
            f"High cardinality features (#unique classes > {self.cardinality_threshold}):"
            f" ({len(high_cardinality_features)}) -  {high_cardinality_features}"
        )
        self.logger.info(f"Numeric features: ({len(numeric_features)}) - {numeric_features}")

        self.logger.info(f"Encoder feature map: \n{json.dumps(feature_encoder_map, indent=4)}")
