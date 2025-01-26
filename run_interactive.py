# %%
# ruff: noqa: ERA001 B018
import logging

from explainerdashboard import ExplainerDashboard

from ml_garden import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# %%
# Path to your experiment configuration json file
# json_path = "examples/ames_housing/configs/1_ames_housing_baseline.json"
# json_path = "examples/ames_housing/configs/2_ames_housing_hp_tuning.json"
# json_path = "examples/ames_housing/configs/3_ames_housing_hp_tuned.json"
json_path = "examples/ames_housing/configs/4_ames_housing_autogluon.json"


# %%
# Load and run the pipeline using the provided JSON file path
pipeline = Pipeline.from_json(json_path)
data = pipeline.train()

# %%
# Load and run the pipeline using the provided JSON file path
pipeline = Pipeline.from_json(json_path)
data = pipeline.predict()

# %%
data.predictions

# %%
ExplainerDashboard(explainer=data.explainer).run()

# %%
data.model.model.leaderboard()
