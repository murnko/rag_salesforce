import datetime
import hashlib
import io
import logging
import os
import pickle
import tempfile
import traceback
import zipfile
from pathlib import Path
from typing import List

import faiss
import numpy as np
from langchain_core.documents import Document
from pathvalidate import sanitize_filename

from data_ingestion.chunks_schema import Chunk, ChunkMetadata
from data_ingestion.document_chunker import DocSplitter  # Adjust import as needed
from data_ingestion.loaders import EnhancedPDFLoader
from data_ingestion.vector_handlers import VectorStoreInterface

# initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocsLoader:
    """
    Document loading class that handles zipped files via streaming.

    Load documents from zip, chunk text and compute embeddings.
    """

    def __init__(
        self,
        text_splitter: DocSplitter,
        vector_store: VectorStoreInterface,
        embedding_model="text-embedding-3-small",
    ):
        self.text_splitter = text_splitter
        self.embedding_model = embedding_model
        self.vector_store = vector_store

        self.all_metadata = [
            "source_doc",
            "source_path",
            "doc_hash",
            "source_sanitized",
            "num_pages",
        ]

        self.extenstions_loaders = {
            "pdf": (
                EnhancedPDFLoader,
                {"convert_to_md": True, "extract_images": True},
            ),
        }

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename using pathvalidate library to ensure cross-platform compatibility."""
        return sanitize_filename(os.path.basename(filename))

    def _load_zip_files(self, zip_path: str):
        """Load documents from zip file."""
        docs = []
        project_dir = os.getcwd()
        tmp_extract_dir = os.path.join(project_dir, "tmp_files")
        os.makedirs(tmp_extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as archive:
            for file_info in archive.infolist():
                if file_info.is_dir():
                    continue
                if "__MACOSX" in file_info.filename or file_info.filename.startswith(
                    "."
                ):  # Skip artifacts
                    continue

                file_extension = file_info.filename.split(".")[-1]
                if file_extension not in self.extenstions_loaders:
                    continue

                with archive.open(file_info) as file:
                    logger.info(f"Processing {file_info.filename}")
                    try:
                        loader_class, loader_kwargs = self.extenstions_loaders[
                            file_extension
                        ]
                        file_content = file.read()

                        # Create temporary file with original extension
                        with tempfile.NamedTemporaryFile(
                            suffix=file_extension, delete=False, dir=tmp_extract_dir
                        ) as temp_file:
                            temp_file_path = temp_file.name
                            temp_file.write(file_content)

                        try:
                            loader = loader_class(temp_file_path, **loader_kwargs)
                            doc = loader.load()
                            doc[0].metadata["source_path"] = file_info.filename
                            doc[0].metadata["source_doc"] = Path(
                                file_info.filename
                            ).name
                            doc[0].metadata["source_sanitized"] = (
                                self._sanitize_filename(file_info.filename)
                            )

                            doc[0].metadata["doc_hash"] = hashlib.sha256(
                                file_content
                            ).hexdigest()
                            docs.append(doc[0])
                        finally:
                            # Clean up temporary file
                            Path(temp_file_path).unlink()
                    except Exception as e:
                        logger.warning(f"Failed to load {file_info.filename}: {e}")
                        logger.warning(traceback.format_exc())
                        continue
        return docs

    def _chunk_docs(self, docs):
        """Chunk documents."""
        all_chunks = []

        for doc in docs:
            try:
                logger.info(f"Processing {doc.metadata.get('source_file', 'unknown')}")

                chunks = self.text_splitter.split_text(doc.page_content)

                print("doc.metadata", doc.metadata)
                for idx, chunk in enumerate(chunks):
                    chunk_id = f"{doc.metadata['source_sanitized']}/{idx}"
                    metadata = {
                        "source_chunk": chunk_id,
                        **{k: doc.metadata.get(k) for k in self.all_metadata},
                    }

                    chunk_obj = Chunk(
                        page_content=chunk,
                        metadata=ChunkMetadata(**metadata),
                    )
                    all_chunks.append(chunk_obj)

            except Exception as e:
                logger.error(
                    f"Failed to process document: {doc.metadata.get('source_file')}",
                    exc_info=True,
                )
                break

        return all_chunks

    def load_and_embed_zip(
        self,
        zip_path: str,
        index_path: str = "faiss.index",
        meta_path: str = "meta.pkl",
    ):
        self.chunks = []

        self.docs = self._load_zip_files(zip_path)
        print("metadata", self.docs[0].metadata)
        self.chunks = self._chunk_docs(self.docs)
        self.vector_store.add_documents(self.chunks)
        self.vector_store.save(index_path)

    def load_from_disk(
        self, index_path: str = "faiss.index", meta_path: str = "meta.pkl"
    ):
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            raise FileNotFoundError(f"Index file {index_path} not found.")

        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
                self.chunks = meta["chunks"]
                self.zip_path = meta.get("zip_path")
        else:
            raise FileNotFoundError(f"Metadata file {meta_path} not found.")

        return self.index, self.chunks
