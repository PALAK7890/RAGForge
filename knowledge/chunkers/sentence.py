"""
Sentence-aware text chunker implementation.
"""

import re
from typing import Dict, List

from knowledge.chunkers.base import BaseChunker, DocumentChunk
from knowledge.loaders.base import LoadedDocument


class SentenceChunker(BaseChunker):
    """Splits documents at sentence boundaries while keeping chunks under chunk_size."""

    def chunk(self, docs: List[LoadedDocument], chunk_size: int, chunk_overlap: int) -> List[DocumentChunk]:
        """
        Chunks documents by grouping sentences up to chunk_size characters.
        
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
        
        # Split on sentence ending punctuation followed by spaces
        sentence_end_re = re.compile(r'(?<=[.!?])\s+')

        for doc in docs:
            text = doc.text.strip()
            if not text:
                continue

            doc_id = doc.document_id
            if doc_id not in running_indices:
                running_indices[doc_id] = 0

            sentences = [s.strip() for s in sentence_end_re.split(text) if s.strip()]
            if not sentences:
                continue

            buffer: List[str] = []
            buffer_len = 0

            for sent in sentences:
                sent_len = len(sent)

                # Fallback: if a single sentence is larger than chunk_size,
                # split it character-wise like the fixed-size chunker.
                if sent_len > chunk_size:
                    if buffer:
                        chunk_text = " ".join(buffer)
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
                        buffer = []
                        buffer_len = 0

                    stride = chunk_size - chunk_overlap
                    start = 0
                    while start < sent_len:
                        end = min(start + chunk_size, sent_len)
                        chunk_text = sent[start:end]
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
                        if end == sent_len:
                            break
                        start += stride
                    continue

                # Normal case: check if adding this sentence exceeds chunk_size
                space_len = 1 if buffer else 0
                if buffer_len + space_len + sent_len > chunk_size:
                    # Flush the current buffer
                    chunk_text = " ".join(buffer)
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

                    # Shift buffer to apply sentence-aware overlap
                    while buffer:
                        current_joined_len = sum(len(s) for s in buffer) + (len(buffer) - 1)
                        if current_joined_len <= chunk_overlap:
                            break
                        buffer.pop(0)

                    buffer_len = sum(len(s) for s in buffer) + (len(buffer) - 1) if buffer else 0

                buffer.append(sent)
                space_len = 1 if len(buffer) > 1 else 0
                buffer_len += space_len + sent_len

            if buffer:
                chunk_text = " ".join(buffer)
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
