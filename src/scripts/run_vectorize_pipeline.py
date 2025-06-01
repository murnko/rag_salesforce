import logging

from scripts.steps.download_step import run_download
from scripts.steps.load_step import run_load

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    docs_path = run_download()
    run_load(docs_path)
    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
