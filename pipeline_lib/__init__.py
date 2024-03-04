from .core.pipeline import Pipeline

Pipeline.auto_register_steps_from_package("pipeline_lib.core.steps")
Pipeline.auto_register_steps_from_package("pipeline_lib.implementation.tabular.xgboost")
