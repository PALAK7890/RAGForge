"""
Document loader router. Matches file extensions to appropriate loaders
and provides safe loading wrappers to handle errors gracefully.
"""

import logging
from pathlib import Path
from typing import Dict, List, Type

from knowledge.loaders.base import BaseLoader, LoadedDocument
from knowledge.loaders.docx import DocxLoader
from knowledge.loaders.html_loader import HtmlLoader
from knowledge.loaders.pdf import PdfLoader
from knowledge.loaders.txt import TxtLoader

logger = logging.getLogger(__name__)


class LoaderRouter:
    """Routes files to the appropriate document loader based on extension."""

    def __init__(self) -> None:
        self.loaders: Dict[str, Type[BaseLoader]] = {
            ".txt": TxtLoader,
            ".md": TxtLoader,
            ".pdf": PdfLoader,
            ".docx": DocxLoader,
            ".html": HtmlLoader,
            ".htm": HtmlLoader,
        }

    def get_loader(self, path: Path) -> BaseLoader:
        """
        Retrieves the appropriate loader for the given file path.
        
        Args:
            path: Path to the document.
            
        Returns:
            An instantiated loader matching the file's extension.
            
        Raises:
            ValueError: If the file extension is not supported.
        """
        ext = path.suffix.lower()
        if ext not in self.loaders:
            raise ValueError(f"Unsupported file format: {ext} for file {path.name}")
        return self.loaders[ext]()


def load_file_safely(path: Path) -> List[LoadedDocument]:
    """
    Loads a document using the appropriate loader, catching and logging any errors
    to prevent the application from crashing.
    
    Args:
        path: Path to the document file.
        
    Returns:
        A list of LoadedDocument instances, or an empty list if loading fails.
    """
    try:
        router = LoaderRouter()
        loader = router.get_loader(path)
        return loader.load(path)
    except Exception as e:
        logger.warning("Skipping file %s: failed to load or parse. Reason: %s", path, str(e))
        return []
