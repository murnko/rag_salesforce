from typing import Union

from langchain.schema import Document
from pydantic import BaseModel, field_validator, model_serializer


class ChunkMetadata(BaseModel):
    """Model containing metadata information about the chunk and document (file) from which it originates."""

    source_doc: str
    """The source filename."""

    source_path: str
    """Relative path to the document (file) with extension from which the chunk originates."""

    doc_hash: str
    """Hash of the document, it is computed based on the content. It is used to identify content has changed."""

    source_sanitized: str
    """The sanitized 'source_doc' value, safe for use in file systems and URLs."""

    source_chunk: str
    """The chunk id."""

    num_pages: int
    """The number of pages in the document."""

    @classmethod
    @field_validator("source_doc")
    def validate_source_doc(cls, value) -> str:
        """Validate that 'source_doc' is not an empty string.

        Args:
            value: The value of 'source_doc' field.

        Returns:
            The value of 'source_doc' field if it is not empty.
        """
        if not value:
            raise ValueError(
                "Field source_doc of ChunkMetadata class cannot be empty string"
            )
        return value


class Chunk(Document):
    """Represents a vector-storable chunk with validated metadata."""

    def __init__(self, page_content: str, metadata: ChunkMetadata, **kwargs):
        # Convert to dict for LangChain
        super().__init__(
            page_content=page_content, metadata=metadata.model_dump(), **kwargs
        )
        self._metadata_obj = metadata

    @classmethod
    def from_chunk_metadata(cls, page_content: str, metadata_obj: ChunkMetadata):
        return cls(page_content=page_content, metadata=metadata_obj)
