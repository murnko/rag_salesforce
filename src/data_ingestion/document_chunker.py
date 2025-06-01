from pathlib import Path
from typing import List, Union

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from data_ingestion.loaders import EnhancedPDFLoader


class DocSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = ["\n\n", "\n", " ", ""],
    ):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators
        )

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        return self.text_splitter.split_text(text)
