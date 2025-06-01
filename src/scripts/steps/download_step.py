from config import settings
from data_ingestion.download_data import download_zip_if_needed


def run_download(url: str = None, output_path: str = None) -> str:
    """Download transcripts zip file and return the path"""
    if url is None:
        url = settings.TRANSCRIPT_ZIP_URL
    if output_path is None:
        output_path = settings.DATA_DIR + "transcripts.zip"

    download_zip_if_needed(url, output_path)
    return output_path


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--url", type=str, default=settings.TRANSCRIPT_ZIP_URL)
#     parser.add_argument(
#         "--output_path", type=str, default=settings.DATA_DIR + "transcripts.zip"
#     )

#     args = parser.parse_args()

#     run_download(args.url, args.output_path)


# if __name__ == "__main__":
#     main()
