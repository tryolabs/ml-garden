import pandas as pd
import xgboost as xgb

from pipeline_lib.core.model import Model


class XGBoostModel(Model):
    def __init__(self, **params):
        self.model = xgb.XGBRegressor(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None, verbose=True) -> None:
        self.model.fit(X, y, eval_set=eval_set, verbose=verbose)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.model.predict(X)
