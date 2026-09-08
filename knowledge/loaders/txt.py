"""
Loader for plain text and Markdown files.
"""

from pathlib import Path
from typing import List

from knowledge.loaders.base import BaseLoader, LoadedDocument, compute_document_id


class TxtLoader(BaseLoader):
    """Parses plain text (.txt) and Markdown (.md) documents."""

    def load(self, path: Path) -> List[LoadedDocument]:
        """
        Loads a plain text or Markdown file.
        
        Args:
            path: Path to the text or markdown file.
            
        Returns:
            A list containing a single LoadedDocument.
        """
        text = path.read_text(encoding="utf-8").strip()
        doc_id = compute_document_id(path)
        fmt = path.suffix.lstrip(".").lower()
        
        return [
            LoadedDocument(
                text=text,
                source_path=str(path.resolve()),
                document_id=doc_id,
                page_number=None,
                metadata={"format": fmt}
            )
        ]
