# %%
import logging
import time
import zipfile
from os import unlink

import pandas as pd
import xgboost as xgb

from pipeline_lib.core.model import Model
from pipeline_lib.utils import compression_utils

logger = logging.getLogger(__file__)


# %%
# Function to save and compress an XGBoost model
def save_and_compress_xgb_booster(
    model: xgb.Booster, filename: str, compression=zipfile.ZIP_BZIP2, compresslevel: int = 9
) -> None:
    """
    Saves an XGBoost model into a compressed file using BZ2 compression.

    Parameters
    ----------
    model : xgb.Booster
        The XGBoost model to save.
    filename : str
        The name of the file where to store the model. File extension should be .ubj or .json.
        Use ".ubj" extension for Binary JSON format (more efficient, non-human-readable) or ".json"
        extension for JSON format.
    compresslevel : int, optional
        Compression level (1-9) to use, higher is more compression. , by default 9
    """

    # Save model directly to a UBJSON file
    start_time = time.time()
    model.save_model(filename)
    logger.debug(f"Model saved in {time.time() - start_time:.2f} seconds.")

    start_time = time.time()
    compression_utils.compress_zipfile(
        filename,
        compression=compression,
        compresslevel=compresslevel,
        delete_uncompressed=True,
    )
    logger.debug(f"Model bzip2 compressed in {time.time() - start_time:.2f} seconds.")


def load_compressed_xgb_booster(filename: str) -> xgb.Booster:
    """
    Loads an XGBoost model from a BZ2 compressed UBJSON file.

    Parameters
    ----------

    filename : str
        The name of the file where the model is stored. File extension should be .ubj or .json,
        the .bz2 extension will be appended automatically

    Returns
    -------
    xgb.Booster
        The loaded XGBoost model.
    """
    compression_utils.decompress_zipfile(filename)

    # Load the model from the temporary UBJSON file
    model = xgb.Booster()
    model.load_model(filename)
    unlink(filename)  # Delete the temporary UBJSON file
    return model


class XGBoostModel(Model):
    def __init__(self, **params):
        self.model = xgb.XGBRegressor(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None, verbose=True) -> None:
        self.model.fit(X, y, eval_set=eval_set, verbose=verbose)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.model.predict(X)
