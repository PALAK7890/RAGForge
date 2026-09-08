"""
Fixed-size text chunker implementation.
"""

from typing import Dict, List

from knowledge.chunkers.base import BaseChunker, DocumentChunk
from knowledge.loaders.base import LoadedDocument


class FixedSizeChunker(BaseChunker):
    """Splits documents into fixed-size character sequences with overlap."""

    def chunk(self, docs: List[LoadedDocument], chunk_size: int, chunk_overlap: int) -> List[DocumentChunk]:
        """
        Chunks documents by slicing them at fixed character indices.
        
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
        # Keep track of running chunk index per unique document_id to avoid ID collisions
        running_indices: Dict[str, int] = {}

        for doc in docs:
            text = doc.text.strip()
            if not text:
                continue

            doc_id = doc.document_id
            if doc_id not in running_indices:
                running_indices[doc_id] = 0

            text_len = len(text)
            stride = chunk_size - chunk_overlap

            # If the text is shorter than chunk_size, output it directly
            if text_len <= chunk_size:
                chunk_index = running_indices[doc_id]
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc_id}_{chunk_index}",
                        text=text,
                        document_id=doc_id,
                        source_path=doc.source_path,
                        page_number=doc.page_number,
                        chunk_index=chunk_index,
                        metadata=doc.metadata.copy()
                    )
                )
                running_indices[doc_id] += 1
                continue

            start = 0
            while start < text_len:
                end = min(start + chunk_size, text_len)
                chunk_text = text[start:end]

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

                if end == text_len:
                    break

                start += stride

        return chunks
