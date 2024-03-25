from .core.pipeline import Pipeline

Pipeline.step_registry.auto_register_steps_from_package("pipeline_lib.core.steps")
Pipeline.model_registry.auto_register_models_from_package(
    "pipeline_lib.implementation.tabular.xgboost"
)
