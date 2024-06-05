from .core.pipeline import Pipeline

Pipeline.step_registry.auto_register_steps_from_package("ml_garden.core.steps")
Pipeline.model_registry.auto_register_models_from_package(
    "ml_garden.implementation.tabular.xgboost"
)
