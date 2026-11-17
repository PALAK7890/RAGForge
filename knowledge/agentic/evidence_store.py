"""
Evidence store managing accumulated retrieval evidence, deduplication, provenance, and citation verification.
"""

import re
from typing import Any, Dict, List, Optional

from knowledge.agentic.schemas import EvidenceRecord, SourceReference


class EvidenceStore:
    """
    In-memory stateful repository of evidence gathered across all agentic retrieval actions.
    Preserves document provenance, prevents duplicate chunk explosion, and verifies citations.
    """

    def __init__(self) -> None:
        self._records: Dict[str, EvidenceRecord] = {}

    def add(self, record: EvidenceRecord) -> str:
        """Add an evidence record to the store and return its evidence_id."""
        self._records[record.evidence_id] = record
        return record.evidence_id

    def add_many(self, records: List[EvidenceRecord]) -> List[str]:
        """Add multiple records, returning their IDs."""
        return [self.add(r) for r in records]

    def get(self, evidence_id: str) -> Optional[EvidenceRecord]:
        """Retrieve a specific record by ID."""
        return self._records.get(evidence_id)

    def get_all(self) -> List[EvidenceRecord]:
        """Return all stored records."""
        return list(self._records.values())

    def __len__(self) -> int:
        return len(self._records)

    def clear(self) -> None:
        """Clear all records from the store."""
        self._records.clear()

    def filter_by_document(self, document_id: str) -> List[EvidenceRecord]:
        """Return records originating from a specific document name or ID."""
        doc_lower = document_id.lower()
        return [
            r for r in self._records.values()
            if doc_lower in r.document_id.lower() or doc_lower in r.source_path.lower()
        ]

    def deduplicate(self) -> int:
        """
        Deduplicate stored evidence records by chunk_id or exact text matching.
        When duplicates occur, retains the one with the highest rerank_score or hybrid_score.
        Returns the number of removed redundant records.
        """
        seen_keys: Dict[str, str] = {}  # key -> evidence_id to keep
        to_remove: List[str] = []

        for eid, record in self._records.items():
            # Identity key: chunk_id if present, else normalized text snippet
            key = record.chunk_id if record.chunk_id else record.text.strip()[:100]

            if key not in seen_keys:
                seen_keys[key] = eid
            else:
                existing_eid = seen_keys[key]
                existing_record = self._records[existing_eid]

                # Compare scores: prioritize higher rerank_score, then hybrid_score
                curr_score = (record.rerank_score or 0.0) + (record.hybrid_score or 0.0)
                exist_score = (existing_record.rerank_score or 0.0) + (existing_record.hybrid_score or 0.0)

                if curr_score > exist_score:
                    # Replace existing with current
                    to_remove.append(existing_eid)
                    seen_keys[key] = eid
                else:
                    to_remove.append(eid)

        for eid in to_remove:
            if eid in self._records:
                del self._records[eid]

        return len(to_remove)

    def generate_citations(self) -> List[Dict[str, Any]]:
        """
        Generate structured citation descriptors for all stored evidence.
        """
        citations = []
        for r in self._records.values():
            citation_label = f"[{r.document_id}"
            if r.page_number is not None:
                citation_label += f", p.{r.page_number}"
            if r.char_start is not None and r.char_end is not None:
                citation_label += f", chars {r.char_start}-{r.char_end}"
            citation_label += "]"

            citations.append({
                "evidence_id": r.evidence_id,
                "label": citation_label,
                "document_id": r.document_id,
                "source_path": r.source_path,
                "page_number": r.page_number,
                "char_start": r.char_start,
                "char_end": r.char_end,
                "chunk_id": r.chunk_id,
                "snippet": r.text[:200] + "..." if len(r.text) > 200 else r.text,
            })
        return citations

    def validate_citations(self, text: str) -> Dict[str, Any]:
        """
        Verify whether in-text citation keys or document references in the generated answer
        correspond to real, retrieved evidence in the store.
        """
        text = str(text or "")
        valid_doc_names = {r.document_id.lower() for r in self._records.values() if r.document_id}
        valid_evidence_ids = set(self._records.keys())

        # Match patterns like [doc.pdf], [doc.pdf, p.1], [evidence:abc12345], [abc12345]
        bracket_citations = re.findall(r"\[([^\]]+)\]", text)
        verified_citations = []
        unverified_citations = []

        for cite in bracket_citations:
            cite_clean = cite.strip()
            # Check if citation refers to an evidence ID
            if cite_clean in valid_evidence_ids:
                verified_citations.append({"type": "evidence_id", "reference": cite_clean})
                continue

            # Check if citation refers to a document name
            matched_doc = any(doc_name in cite_clean.lower() for doc_name in valid_doc_names)
            if matched_doc:
                verified_citations.append({"type": "document_name", "reference": cite_clean})
            else:
                unverified_citations.append({"type": "unknown", "reference": cite_clean})

        return {
            "total_cited": len(bracket_citations),
            "verified_count": len(verified_citations),
            "unverified_count": len(unverified_citations),
            "verified": verified_citations,
            "unverified": unverified_citations,
            "citation_integrity": (
                len(verified_citations) / len(bracket_citations) if bracket_citations else 1.0
            ),
        }

    def format_context_for_prompt(self, max_tokens: int = 4000) -> str:
        """
        Render evidence store content formatted for LLM context injection.
        """
        blocks = []
        char_budget = max_tokens * 4
        current_chars = 0

        # Sort by relevance score desc
        sorted_records = sorted(
            self._records.values(),
            key=lambda r: (r.rerank_score or 0.0, r.hybrid_score or 0.0),
            reverse=True,
        )

        for r in sorted_records:
            header = f"--- [Evidence: {r.evidence_id} | Document: {r.document_id}"
            if r.page_number is not None:
                header += f" | Page: {r.page_number}"
            header += " ---"

            block = f"{header}\n{r.text}\n"
            if current_chars + len(block) > char_budget:
                break
            blocks.append(block)
            current_chars += len(block)

        return "\n".join(blocks)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize evidence store to dictionary."""
        return {eid: rec.model_dump() for eid, rec in self._records.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceStore":
        """Deserialize evidence store from dictionary."""
        store = cls()
        for eid, rec_data in data.items():
            store._records[eid] = EvidenceRecord(**rec_data)
        return store
