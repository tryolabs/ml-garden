# %%
import argparse
import logging

import pandas as pd

from pipeline_lib import Pipeline

pd.options.display.max_rows = 100


logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger()

# Create an argument parser
parser = argparse.ArgumentParser(description="Run pipeline from JSON file")
parser.add_argument("json_path", help="Path to the JSON file")

# Add mutually exclusive group for train and predict arguments
mode_group = parser.add_mutually_exclusive_group()
mode_group.add_argument(
    "--train", action="store_true", help="Run the pipeline in train mode (default)"
)
mode_group.add_argument("--predict", action="store_true", help="Run the pipeline in predict mode")

# Parse the command-line arguments
args = parser.parse_args()

# Get the JSON file path from the command-line argument
json_path = args.json_path

# Determine the mode based on the provided arguments
is_train = not args.predict

# Load and run the pipeline using the provided JSON file path
data = Pipeline.from_json(json_path).run(is_train=is_train)

# %%
