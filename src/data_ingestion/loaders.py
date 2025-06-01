import logging
import os
import tempfile

import fitz
import pymupdf4llm
import pypandoc
from langchain.document_loaders import PyMuPDFLoader
from langchain.schema import Document

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_page_count(file_path):
    with fitz.open(file_path) as doc:
        return doc.page_count


class EnhancedPDFLoader(PyMuPDFLoader):
    """Enhanced loader with optional image extraction and markdown conversion."""

    def __init__(
        self, file_path: str, convert_to_md: bool = False, extract_images: bool = False
    ):
        file_path = str(file_path)
        super().__init__(file_path)
        self._file_path = file_path
        self._convert_to_md = convert_to_md
        self._extract_images = extract_images

    def load(self):
        documents = None
        if self._convert_to_md:
            try:
                documents = self._convert_pdf_to_markdown()
                logger.info("PDF successfully converted to markdown.")
            except Exception as error:
                logger.warning(
                    "Markdown conversion failed; using fallback loader.", exc_info=True
                )

        if documents is None:
            documents = super().load()

        return documents

    def _convert_pdf_to_markdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            media_dir = os.path.join(tmpdir, "media")
            gfm = pymupdf4llm.to_markdown(
                self._file_path,
                write_images=self._extract_images,
                image_path=media_dir,
            )
            markdown = pypandoc.convert_text(gfm, "markdown", format="gfm")
            doc = Document(page_content=markdown)
            doc.metadata.update(
                {
                    "converted_to": "markdown",
                    "num_pages": get_page_count(self._file_path),
                }
            )
            return [doc]
