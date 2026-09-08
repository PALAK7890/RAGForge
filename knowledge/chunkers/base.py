"""
Base schema and interface for text chunking strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from knowledge.loaders.base import LoadedDocument


class DocumentChunk(BaseModel):
    """Represents a single processed chunk of text from a LoadedDocument."""
    
    chunk_id: str = Field(
        description="Unique identifier for the chunk, formatted as {document_id}_{chunk_index}"
    )
    text: str = Field(description="The sliced content of this text chunk")
    document_id: str = Field(description="Reference to the parent document's unique ID")
    source_path: str = Field(description="The source file path of the parent document")
    page_number: Optional[int] = Field(
        default=None,
        description="The source page number of this chunk (None if unpaginated)"
    )
    chunk_index: int = Field(description="0-indexed position of this chunk within the document")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible metadata inherited or generated for this chunk"
    )


class BaseChunker(ABC):
    """Abstract Base Class for all text chunkers."""

    @abstractmethod
    def chunk(self, docs: List[LoadedDocument], chunk_size: int, chunk_overlap: int) -> List[DocumentChunk]:
        """
        Chunks a list of LoadedDocuments into a list of smaller DocumentChunks.
        
        Note:
            We execute chunking page-by-page (i.e. separately for each LoadedDocument).
            This guarantees that chunk boundaries never cross physical pages, preserving 
            100% accurate page-level citations at retrieval time.
            
        Args:
            docs: The list of LoadedDocuments (pages/files).
            chunk_size: Maximum character length of each chunk.
            chunk_overlap: Overlapping character count between consecutive chunks.
            
        Returns:
            A list of DocumentChunk instances.
        """
        pass
