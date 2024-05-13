import logging
import sys
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


def get_data(url: str, attempts: int = 4, wait: int = 3, wait_multiple: int = 2) -> bytes:
    """Acquires data from URL

    Args:
        url: The URL from which to acquire data
        attempts: Number of attempts to try downloading the data
        wait: Time to wait between download attempts
        wait_multiple: Factor by which to multiply the wait time between attempts


    Returns:
        bytes: The content of the downloaded data
    """
    for _ in range(attempts):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise an exception for non-200 status codes
            return response.content
        except requests.RequestException as e:
            logger.error(
                "Error occurred while downloading data from %s: %s", url, e)
            logger.info("Retrying in %s seconds...", wait)
            time.sleep(wait)
            wait *= wait_multiple
    raise IOError(
        f"Failed to download data from {url} after {attempts} attempts.")


def write_data(data: bytes, save_path: Path) -> None:
    """Writes data to the specified file path

    Args:
        data: The data to write
        save_path: The file path to write the data to
    """
    with open(save_path, "wb") as f:
        f.write(data)


def acquire_data(url: str, save_path: Path) -> None:
    """Acquires data from the specified URL and saves it to disk

    Args:
        url: The URL from which to acquire the data
        save_path: The local path to save the acquired data
    """
    url_contents = get_data(url)
    try:
        write_data(url_contents, save_path)
        logger.info("Data written to %s", save_path)
    except FileNotFoundError:
        logger.error(
            "Please provide a valid file location to save dataset to.")
        sys.exit(1)
    except Exception as e:
        logger.error(
            "Error occurred while trying to write dataset to file: %s", e)
        sys.exit(1)
