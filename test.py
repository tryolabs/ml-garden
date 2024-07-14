# %%
import logging

import pandas as pd

from ml_garden import Pipeline

logging.basicConfig(level=logging.INFO)

# pipeline = Pipeline.from_json("tests/data/test.json")
# pipeline = Pipeline.from_json("examples/delay/configs/autogluon.json")
pipeline = Pipeline.from_json("examples/delay/configs/base.json")

# %%
data = pipeline.train()
# %%
data.X_train
# %%
from explainerdashboard import ExplainerDashboard

# Create a dashboard
dashboard = ExplainerDashboard(data.explainer)

# Run the dashboard
dashboard.run()
# %%
data = pipeline.predict()
# %%
data.predictions
# %%
data.feature_importance
# %%
df = pd.read_csv("examples/delay/data/train.csv")
df

# %%
data.X_train
# %%
df.iloc[1:]
# %%
data.predictions
# %%
# data._generate_step_dtypes


import logging

# %%
from pipeline_lib import Pipeline

logging.basicConfig(level=logging.INFO)

pipeline = Pipeline.from_json("tests/data/test.json")
data = pipeline.train()
data._generate_step_dtypes
data.target
# %%

data = pipeline.predict()

# %%
