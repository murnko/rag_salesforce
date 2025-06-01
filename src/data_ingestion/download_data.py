# Download data

import logging
import os
import zipfile

import requests

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def download_zip_if_needed(url: str, save_path: str = "data/transcripts.zip") -> str:
    if os.path.isfile(save_path):
        logger.info(f"Using cached ZIP at: {save_path}")
        return save_path

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    logger.info(f"Fetching ZIP from: {url}")

    try:
        with requests.get(url, stream=True, timeout=15) as r:
            r.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        logger.info("Download successful.")
        return save_path
    except requests.RequestException as e:
        logger.error(f"Download failed: {e}")
        raise

def unzip_file(zip_path: str, output_dir: str):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
