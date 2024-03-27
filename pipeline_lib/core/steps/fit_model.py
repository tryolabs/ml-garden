import json
from typing import Optional, Type

import optuna

from pipeline_lib.core import DataContainer
from pipeline_lib.core.model import Model
from pipeline_lib.core.steps.base import PipelineStep
from pipeline_lib.core.utils import calculate_error


class FitModelStep(PipelineStep):
    """Fit the model."""

    used_for_prediction = False
    used_for_training = True

    def __init__(
        self,
        model_class: Type[Model],
        target: str,
        model_params: Optional[dict] = None,
        drop_columns: Optional[list[str]] = None,
        optuna_params: Optional[dict] = None,
        search_space: Optional[dict] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Initialize FitModelStep."""
        super().__init__()
        self.init_logger()
        self.model_class = model_class
        self.target = target
        self.model_params = model_params or {}
        self.drop_columns = drop_columns or []
        self.optuna_params = optuna_params
        self.search_space = search_space or {}
        self.save_path = save_path

    def execute(self, data: DataContainer) -> DataContainer:
        """Execute the step."""
        self.logger.info(f"Fitting the {self.model_class.__name__} model")

        df_train = data.train
        df_valid = data.validation

        if self.drop_columns:
            df_train = df_train.drop(columns=self.drop_columns)
            df_valid = df_valid.drop(columns=self.drop_columns)

        X_train = df_train.drop(columns=[self.target])
        y_train = df_train[self.target]

        X_valid = df_valid.drop(columns=[self.target])
        y_valid = df_valid[self.target]

        params = self.model_params

        if self.optuna_params:
            objective_metric = self.optuna_params.get("objective_metric", "mean_absolute_error")
            params = self.optimize_with_optuna(X_train, y_train, X_valid, y_valid, objective_metric)
            data.tuning_params = params
            self.logger.info(
                "Finished optimizing the model with Optuna. Using the best parameters to fit the"
                " model."
            )
            # add model_params to params
            params.update(self.model_params)

        self.logger.info(
            f"Fitting the model with the following parameters: \n {json.dumps(params, indent=4)}"
        )
        self.model = self.model_class(**params)

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=True,
        )

        data.model = self.model
        data.target = self.target
        data._drop_columns = self.drop_columns

        if self.save_path:
            self.logger.info(f"Saving the model to {self.save_path}")
            self.model.save(self.save_path)

        return data

    def optimize_with_optuna(self, X_train, y_train, X_valid, y_valid, objective_metric: str):
        def objective(trial):
            param = {}
            for key, value in self.search_space.items():
                if isinstance(value, dict):
                    suggest_func = getattr(trial, f"suggest_{value['type']}")
                    kwargs = value.get("kwargs", {})
                    param[key] = suggest_func(key, *value["args"], **kwargs)
                else:
                    param[key] = value

            # add model_params to params
            param.update(self.model_params)

            model = self.model_class(**param)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=True,
            )
            preds = model.predict(X_valid)

            error = calculate_error(y_valid, preds, objective_metric)

            return error

        def optuna_logging_callback(study, trial):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                self.logger.info(
                    f"Trial {trial.number} finished with value: {trial.value} and parameters:"
                    f" {trial.params}. Best is trial {study.best_trial.number} with value:"
                    f" {study.best_value}."
                )

        if not self.optuna_params:
            raise ValueError("Optuna parameters are not provided.")

        optuna_trials = self.optuna_params.get("trials", 20)

        self.logger.info(f"Optimizing XGBoost hyperparameters with {optuna_trials} trials.")

        study_name = self.optuna_params.get("study_name")

        if not study_name:
            raise ValueError("Study name is not provided in the optuna parameters.")

        load_if_exists = self.optuna_params.get("load_if_exists", False)
        if load_if_exists:
            self.logger.info("Loading existing study if it exists. It will overwrite the study.")

        storage = self.optuna_params.get("storage", "sqlite:///db.sqlite3")

        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            storage=storage,
            load_if_exists=load_if_exists,
        )

        study.optimize(objective, n_trials=optuna_trials, callbacks=[optuna_logging_callback])

        best_params = study.best_params
        self.logger.info(f"Best parameters found by Optuna: \n{json.dumps(best_params, indent=4)}")
        return best_params
