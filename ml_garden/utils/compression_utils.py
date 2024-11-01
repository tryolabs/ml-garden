import logging
import time
import zipfile
from pathlib import Path

import xgboost as xgb

logger = logging.getLogger(__name__)


def compress_zipfile(
    filename: str,
    compression: int = zipfile.ZIP_BZIP2,
    compresslevel: int = 9,
    *,
    delete_uncompressed: bool = False,
) -> None:
    """
    Compress a single file into a .zip file using the algorithm specified by compression parameter.

    If delete_uncompressed is True, it will also delete the original uncompressed file after
    compression.

    Parameters
    ----------
    filename : str
        The name of the file to be compressed
    compression : _type_, optional
        The compression algorithm to be used (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED,
        zipfile.ZIP_BZIP2, or zipfile.ZIP_LZMA).
        In general zipfile.ZIP_LZMA < zipfile.ZIP_BZIP2 < zipfile.ZIP_DEFLATED < zipfile.ZIP_STORED
        both in terms of compressed file size and compute resources needed for
        compression/decompression. BZIP2 is usually a good compromise when looking for smaller sizes
        while DEFLATED is a good choice when looking for compression/decompression speed.
        , by default zipfile.ZIP_LZMA

    compresslevel : int, optional
        The level of compression to be used (1-9), by default 9
    delete_uncompressed : bool, optional
        If True, the original file will be deleted after compression, by default False
    """
    with zipfile.ZipFile(
        filename + ".zip",
        "w",
        compression=compression,
        compresslevel=compresslevel,
    ) as zip_file:
        zip_file.write(filename, arcname=Path(filename).name)

    if delete_uncompressed:
        Path(filename).unlink()


def decompress_zipfile(filename: str) -> None:
    """
    Extract all files contained in a .zip file to the current directory.

    Filename must not contain .zip extension, it will be added automatically by this function

    Parameters
    ----------
    filename : str
        The name of the .zip file to be decompressed, without the .zip extension
    """
    with zipfile.ZipFile(filename + ".zip", "r") as zip_file:
        zip_file.extractall(path=Path(filename).parent)


# Function to save and compress an XGBoost model
def save_and_compress_xgb_booster(
    model: xgb.Booster, filename: str, compression: int = zipfile.ZIP_BZIP2, compresslevel: int = 9
) -> None:
    """
    Save an XGBoost model into a compressed file using BZ2 compression.

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
    logger.debug("Model saved in %.2f seconds.", time.time() - start_time)

    start_time = time.time()
    compress_zipfile(
        filename,
        compression=compression,
        compresslevel=compresslevel,
        delete_uncompressed=True,
    )
    logger.debug("Model bzip2 compressed in %.2f seconds.", time.time() - start_time)


def load_compressed_xgb_booster(filename: str) -> xgb.Booster:
    """
    Load an XGBoost model from a BZ2 compressed UBJSON file.

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
    decompress_zipfile(filename)

    # Load the model from the temporary UBJSON file
    model = xgb.Booster()
    model.load_model(filename)
    Path(filename).unlink()  # Delete the temporary UBJSON file
    return model
