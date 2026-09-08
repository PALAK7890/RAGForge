"""
Base classes and schemas for document loading.
"""

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def compute_document_id(path: Path) -> str:
    """
    Stable ID for a document, used to match chunks back to their source
    during removal/rebuild even if paths are relative vs. absolute.
    """
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]


class LoadedDocument(BaseModel):
    """Represents a single segment or page of a loaded document."""
    text: str = Field(description="The raw text extracted from the document source")
    source_path: str = Field(description="The absolute or relative file path of the source document")
    document_id: str = Field(description="A stable unique identifier for the document")
    page_number: Optional[int] = Field(default=None, description="1-indexed page number if applicable (e.g. for PDFs)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extensible metadata dictionary for additional fields")


class BaseLoader(ABC):
    """Abstract Base Class interface for all document loaders."""

    @abstractmethod
    def load(self, path: Path) -> List[LoadedDocument]:
        """
        Loads and parses a document into a list of LoadedDocument instances.
        
        Args:
            path: Path to the target document.
            
        Returns:
            A list of LoadedDocument objects, e.g., one per page or one for the entire file.
            
        Raises:
            Exception: If file parsing fails or file format is unsupported.
        """
        pass
