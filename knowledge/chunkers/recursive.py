"""
Recursive character text chunker implementation.
"""

from typing import Dict, List, Optional

from knowledge.chunkers.base import BaseChunker, DocumentChunk
from knowledge.loaders.base import LoadedDocument


class RecursiveChunker(BaseChunker):
    """Splits documents recursively using a list of hierarchical separators."""

    def __init__(self, separators: Optional[List[str]] = None) -> None:
        # Default separators: paragraphs, newlines, words, characters
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def chunk(self, docs: List[LoadedDocument], chunk_size: int, chunk_overlap: int) -> List[DocumentChunk]:
        """
        Chunks documents by splitting recursively along hierarchical separators.
        
        Args:
            docs: List of loaded documents to chunk.
            chunk_size: The character size of each chunk.
            chunk_overlap: The character overlap between consecutive chunks.
            
        Returns:
            A list of DocumentChunks.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative integer")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        chunks: List[DocumentChunk] = []
        running_indices: Dict[str, int] = {}

        for doc in docs:
            text = doc.text.strip()
            if not text:
                continue

            doc_id = doc.document_id
            if doc_id not in running_indices:
                running_indices[doc_id] = 0

            # Split the document text recursively
            split_texts = self._split_text(text, self.separators, chunk_size, chunk_overlap)

            for chunk_text in split_texts:
                chunk_text = chunk_text.strip()
                if not chunk_text:
                    continue

                chunk_index = running_indices[doc_id]
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc_id}_{chunk_index}",
                        text=chunk_text,
                        document_id=doc_id,
                        source_path=doc.source_path,
                        page_number=doc.page_number,
                        chunk_index=chunk_index,
                        metadata=doc.metadata.copy()
                    )
                )
                running_indices[doc_id] += 1

        return chunks

    def _split_text(self, text: str, separators: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
        """Recursive helper to split text into chunks of maximum size."""
        if len(text) <= chunk_size:
            return [text]

        if not separators:
            # Fallback character level split if we run out of separators
            return self._character_split(text, chunk_size, chunk_overlap)

        separator = separators[0]
        next_separators = separators[1:]

        # Handle character fallback separator specially since Python doesn't allow splitting by empty string
        if separator == "":
            return self._character_split(text, chunk_size, chunk_overlap)

        parts = text.split(separator)
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for part in parts:
            # Skip empty parts (unless there are no other parts)
            if not part and len(parts) > 1:
                continue

            part_len = len(part)

            # If a single part exceeds chunk_size, recursively split it
            if part_len > chunk_size:
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                    current_chunk = []
                    current_len = 0

                sub_chunks = self._split_text(part, next_separators, chunk_size, chunk_overlap)
                chunks.extend(sub_chunks)
                continue

            # Standard merging logic
            sep_len = len(separator) if current_chunk else 0
            if current_len + sep_len + part_len > chunk_size:
                chunks.append(separator.join(current_chunk))

                # Slide back for overlap: keep last items that fit within chunk_overlap limit
                new_chunk: List[str] = []
                new_len = 0
                for p in reversed(current_chunk):
                    p_sep_len = len(separator) if new_chunk else 0
                    if new_len + p_sep_len + len(p) <= chunk_overlap:
                        new_chunk.insert(0, p)
                        new_len += p_sep_len + len(p)
                    else:
                        break
                current_chunk = new_chunk
                current_len = new_len

            sep_len = len(separator) if current_chunk else 0
            current_chunk.append(part)
            current_len += sep_len + part_len

        if current_chunk:
            chunks.append(separator.join(current_chunk))

        return chunks

    def _character_split(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Slices text into character-wise sequences with overlap."""
        stride = chunk_size - chunk_overlap
        chunks: List[str] = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunks.append(text[start:end])
            if end == text_len:
                break
            start += stride
            
        return chunks
