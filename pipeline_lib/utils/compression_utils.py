import os
import zipfile


def compress_zipfile(
    filename: str,
    compression: int = zipfile.ZIP_BZIP2,
    compresslevel: int = 9,
    delete_uncompressed: bool = False,
) -> None:
    """
    Compress a single file into a .zip file using the algorithm specified by compression parameter
    (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2, or zipfile.ZIP_LZMA) and with the
    specified compression level (1-9).
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
        zip_file.write(filename, arcname=os.path.basename(filename))

    if delete_uncompressed:
        os.unlink(filename)


def decompress_zipfile(filename: str):
    """
    Extract all files contained in a .zip file to the current directory.
    filename must not contain .zip extension, it will be added automatically by this function

    Parameters
    ----------
    filename : str
        The name of the .zip file to be decompressed, without the .zip extension
    """
    with zipfile.ZipFile(filename + ".zip", "r") as zip_file:
        zip_file.extractall(path=os.path.dirname(filename))
