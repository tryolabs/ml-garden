# %%
import logging

from explainerdashboard import ExplainerDashboard

from pipeline_lib import Pipeline

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

# %%
# Path to your experiment configuration json file
json_path = "examples/ames_housing/configs/1_ames_housing_baseline.json"
# json_path = "examples/ames_housing/configs/2_ames_housing_hp_tuning.json"
# json_path = "examples/ames_housing/configs/3_ames_housing_hp_tuned.json"
is_train = True


# %%
# Load and run the pipeline using the provided JSON file path
pipeline = Pipeline.from_json(json_path)
data = pipeline.run(is_train=is_train)


# %%
# %%
# Load and run the pipeline using the provided JSON file path
pipeline = Pipeline.from_json(json_path)
data = pipeline.run(is_train=False)

# %%
data.predictions

# %%
ExplainerDashboard(explainer=data.explainer).run()

# %%
