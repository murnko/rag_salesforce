import argparse

from config import settings
from data_ingestion.docs_loader import DocsLoader
from data_ingestion.document_chunker import DocSplitter
from data_ingestion.vector_handlers import AzureSearchAdapter, FAISSAdapter


def run_load(input_path: str = settings.DATA_DIR + "transcripts.zip"):
    text_splitter = DocSplitter()
    if settings.VECTOR_STORE == "FAISS":
        vector_store = FAISSAdapter()
    elif settings.VECTOR_STORE == "AZURE_SEARCH":
        vector_store = AzureSearchAdapter()
    docs_loader = DocsLoader(text_splitter=text_splitter, vector_store=vector_store)
    docs_loader.load_and_embed_zip(input_path)


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--input_path", type=str, default=settings.DATA_DIR + "transcripts.zip"
#     )
#     args = parser.parse_args()

#     docs_loader = DocsLoader()
#     docs = docs_loader.load(args.input_path)
#     print(docs[0].metadata)


# if __name__ == "__main__":
#     main()
