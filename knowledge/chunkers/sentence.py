"""
Sentence-aware text chunker implementation.
"""

import re
from typing import Dict, List, Optional

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
            A list of DocumentChunks with char_start/char_end set where possible.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative integer")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        chunks: List[DocumentChunk] = []
        running_indices: Dict[str, int] = {}

        # Split on sentence-ending punctuation followed by whitespace
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
            # Cursor advances through `text` so repeated substrings are found correctly
            search_from = 0

            def _flush_buffer(buf: List[str], doc_ref: LoadedDocument, text_ref: str) -> None:
                """Append a chunk for the buffered sentences and advance search_from."""
                nonlocal search_from
                chunk_text = " ".join(buf)
                pos = text_ref.find(chunk_text, search_from)
                if pos != -1:
                    cs: Optional[int] = pos
                    ce: Optional[int] = pos + len(chunk_text)
                    search_from = pos + 1
                else:
                    cs = None
                    ce = None
                target_doc_id = doc_ref.document_id
                chunk_index = running_indices[target_doc_id]
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{target_doc_id}_{chunk_index}",
                        text=chunk_text,
                        document_id=target_doc_id,
                        source_path=doc_ref.source_path,
                        page_number=doc_ref.page_number,
                        chunk_index=chunk_index,
                        metadata=doc_ref.metadata.copy(),
                        char_start=cs,
                        char_end=ce,
                    )
                )
                running_indices[target_doc_id] += 1

            for sent in sentences:
                sent_len = len(sent)

                # Fallback: a single sentence larger than chunk_size → char-split it.
                if sent_len > chunk_size:
                    if buffer:
                        _flush_buffer(buffer, doc, text)
                        buffer = []
                        buffer_len = 0

                    sent_pos = text.find(sent, search_from)
                    stride = chunk_size - chunk_overlap
                    start = 0
                    while start < sent_len:
                        end = min(start + chunk_size, sent_len)
                        chunk_text = sent[start:end]
                        chunk_index = running_indices[doc_id]
                        if sent_pos != -1:
                            cs = sent_pos + start
                            ce = sent_pos + end
                        else:
                            cs = None
                            ce = None
                        chunks.append(
                            DocumentChunk(
                                chunk_id=f"{doc_id}_{chunk_index}",
                                text=chunk_text,
                                document_id=doc_id,
                                source_path=doc.source_path,
                                page_number=doc.page_number,
                                chunk_index=chunk_index,
                                metadata=doc.metadata.copy(),
                                char_start=cs,
                                char_end=ce,
                            )
                        )
                        running_indices[doc_id] += 1
                        if end == sent_len:
                            if sent_pos != -1:
                                search_from = sent_pos + sent_len
                            break
                        start += stride
                    continue

                # Normal path: accumulate sentences into a buffer up to chunk_size
                space_len = 1 if buffer else 0
                if buffer_len + space_len + sent_len > chunk_size:
                    _flush_buffer(buffer, doc, text)

                    # Retain trailing sentences that fit within chunk_overlap
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
                _flush_buffer(buffer, doc, text)

        return chunks
