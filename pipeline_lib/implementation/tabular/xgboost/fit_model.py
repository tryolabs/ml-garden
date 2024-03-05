import time

import optuna
import xgboost as xgb
from optuna.pruners import MedianPruner
from sklearn.metrics import mean_absolute_error

from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps import FitModelStep


class XGBoostFitModelStep(FitModelStep):
    """Fit the model with XGBoost."""

    def execute(self, data: DataContainer) -> DataContainer:
        self.logger.debug("Starting model fitting with XGBoost")

        start_time = time.time()

        model_configs = self.config

        if model_configs is None:
            raise ValueError("No model configs found")

        target = model_configs.get("target")

        if target is None:
            raise ValueError("Target column not found in model_configs.")

        data[DataContainer.TARGET] = target

        df_train = data[DataContainer.TRAIN]
        df_valid = data[DataContainer.VALIDATION]

        drop_columns = model_configs.get("drop_columns")

        if drop_columns:
            df_train = df_train.drop(columns=drop_columns)
            df_valid = df_valid.drop(columns=drop_columns)

        # Prepare the data
        X_train = df_train.drop(columns=[target])
        y_train = df_train[target]

        X_valid = df_valid.drop(columns=[target])
        y_valid = df_valid[target]

        optuna_params = model_configs.get("optuna_params")
        xgb_params = model_configs.get("xgb_params")

        if optuna_params and xgb_params:
            raise ValueError("Both optuna_params and xgb_params are defined. Please choose one.")

        if not optuna_params and not xgb_params:
            raise ValueError(
                "No parameters defined. Please define either optuna_params or xgb_params."
            )

        params = xgb_params

        if optuna_params:
            params = self.optimize_with_optuna(X_train, y_train, X_valid, y_valid, optuna_params)
            data[DataContainer.TUNING_PARAMS] = params

        model = xgb.XGBRegressor(**params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=model_configs.get("early_stopping_rounds", 100),
            verbose=True,
        )

        # Save the model to the data container
        data[DataContainer.MODEL] = model

        importance = model.get_booster().get_score(importance_type="gain")
        importance = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
        data[DataContainer.IMPORTANCE] = importance

        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        self.logger.info(f"XGBoost model fitting took {minutes} minutes and {seconds} seconds.")
        return data

    def optimize_with_optuna(self, X_train, y_train, X_valid, y_valid, optuna_params):
        def objective(trial):
            # Define the search space
            max_depth = optuna_params.get("max_depth", [3, 12])
            eta = optuna_params.get("eta", [1e-8, 1.0])
            subsample = optuna_params.get("subsample", [0.2, 1.0])
            colsample_bytree = optuna_params.get("colsample_bytree", [0.2, 1.0])
            min_child_weight = optuna_params.get("min_child_weight", [1, 10])
            n_estimators = optuna_params.get("n_estimators", [100, 1000])

            param = {
                "verbosity": 0,
                "objective": "reg:squarederror",
                "eval_metric": "mae",
                "n_jobs": -1,
                "max_depth": trial.suggest_int("max_depth", max_depth[0], max_depth[1]),
                "eta": trial.suggest_float("eta", eta[0], eta[1], log=True),
                "subsample": trial.suggest_float("subsample", subsample[0], subsample[1]),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", colsample_bytree[0], colsample_bytree[1]
                ),
                "min_child_weight": trial.suggest_int(
                    "min_child_weight", min_child_weight[0], min_child_weight[1]
                ),
                "n_estimators": trial.suggest_int("n_estimators", n_estimators[0], n_estimators[1]),
            }

            model = xgb.XGBRegressor(**param)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                early_stopping_rounds=optuna_params.get("early_stopping_rounds", 50),
                verbose=True,
            )
            preds = model.predict(X_valid)
            mae = mean_absolute_error(y_valid, preds)
            return mae

        def optuna_logging_callback(study, trial):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                self.logger.info(
                    f"Trial {trial.number} finished with value: {trial.value} and parameters:"
                    f" {trial.params}. Best is trial {study.best_trial.number} with value:"
                    f" {study.best_value}."
                )

        optuna_trials = optuna_params.get("trials", 20)

        self.logger.info(f"Optimizing XGBoost hyperparameters with {optuna_trials} trials.")

        study_name = optuna_params.get("study_name", "xgboost_optimization")
        storage = optuna_params.get("storage", "sqlite:///db.sqlite3")

        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            storage=storage,
            pruner=MedianPruner(),
        )

        study.optimize(objective, n_trials=optuna_trials, callbacks=[optuna_logging_callback])

        best_params = study.best_params
        self.logger.info(f"Best parameters found by Optuna: {best_params}")
        return best_params
