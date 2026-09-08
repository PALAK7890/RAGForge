"""
Loader for HTML documents.
"""

from pathlib import Path
from typing import List

from bs4 import BeautifulSoup

from knowledge.loaders.base import BaseLoader, LoadedDocument, compute_document_id


class HtmlLoader(BaseLoader):
    """Parses HTML documents using BeautifulSoup4."""

    def load(self, path: Path) -> List[LoadedDocument]:
        """
        Loads an HTML file and extracts pure visible text.
        
        Args:
            path: Path to the HTML document.
            
        Returns:
            A list containing a single LoadedDocument.
        """
        html_content = path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Strip script and style elements to avoid indexing non-text content
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
            
        # Get text and clean up excessive newlines/whitespaces
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        cleaned_text = "\n".join(chunk for chunk in chunks if chunk).strip()
        
        doc_id = compute_document_id(path)
        
        return [
            LoadedDocument(
                text=cleaned_text,
                source_path=str(path.resolve()),
                document_id=doc_id,
                page_number=None,
                metadata={"format": "html"}
            )
        ]
