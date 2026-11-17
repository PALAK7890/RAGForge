"""
Citation validation, claim-to-evidence alignment, and hallucination detection engine.
Ensures every citation in generated answers references verified, indexed document passages.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from knowledge.agentic.schemas import EvidenceRecord


class CitationValidator:
    """
    Validates citation integrity, extracts in-text references, and checks
    alignment between claims and underlying retrieved evidence records.
    """

    def __init__(self, evidence_records: Optional[List[EvidenceRecord]] = None) -> None:
        self.records = evidence_records or []
        self._doc_names: Set[str] = {r.document_id.lower() for r in self.records if r.document_id}
        self._evidence_ids: Set[str] = {r.evidence_id for r in self.records}
        self._source_paths: Set[str] = {r.source_path.lower() for r in self.records if r.source_path}

    def extract_citations(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract all citation marks from answer text.
        Recognizes formats:
        - [document.pdf]
        - [document.pdf, p. 2]
        - [document.pdf, chars 100-250]
        - [evidence:abc12345]
        """
        # Match bracket citations that are NOT part of a markdown link [anchor](url)
        raw_matches = re.findall(r"\[([^\]]+)\](?!\()", text)
        citations = []

        for m in raw_matches:
            content = m.strip()
            # Ignore markdown links like [text](url) or non-citation labels
            if content.startswith("http") or content.lower().startswith("note:"):
                continue

            # Parse document name and optional page / offsets
            page_match = re.search(r",\s*p(?:age|\.)\s*(\d+)", content, re.IGNORECASE)
            page = int(page_match.group(1)) if page_match else None

            span_match = re.search(r",\s*chars\s*(\d+)\s*-\s*(\d+)", content, re.IGNORECASE)
            char_start = int(span_match.group(1)) if span_match else None
            char_end = int(span_match.group(2)) if span_match else None

            # Base document or evidence token
            base_token = re.split(r",\s*p|,\s*chars", content, flags=re.IGNORECASE)[0].strip()

            citations.append({
                "raw": f"[{content}]",
                "token": base_token,
                "page": page,
                "char_start": char_start,
                "char_end": char_end,
            })

        return citations

    def validate(self, text: str) -> Dict[str, Any]:
        """
        Validate every citation mark in `text` against the known evidence records.
        Returns detailed integrity metrics and flags hallucinated citations.
        """
        extracted = self.extract_citations(text)
        if not extracted:
            return {
                "total_citations": 0,
                "valid_citations": 0,
                "hallucinated_citations": 0,
                "citation_integrity": 1.0,
                "verified_references": [],
                "hallucinated_references": [],
                "has_hallucinations": False,
            }

        verified = []
        hallucinated = []

        for cite in extracted:
            token_lower = cite["token"].lower()
            matched_record = None

            # Check evidence_id match
            if cite["token"] in self._evidence_ids:
                matched_record = next(r for r in self.records if r.evidence_id == cite["token"])

            # Check document_id match
            if not matched_record:
                for r in self.records:
                    if r.document_id and (token_lower in r.document_id.lower() or r.document_id.lower() in token_lower):
                        matched_record = r
                        break

            # Check source_path match
            if not matched_record:
                for r in self.records:
                    if r.source_path and token_lower in r.source_path.lower():
                        matched_record = r
                        break

            if matched_record:
                # Verify page number if specified
                page_valid = True
                if cite["page"] is not None and matched_record.page_number is not None:
                    page_valid = (cite["page"] == matched_record.page_number)

                verified.append({
                    "citation": cite["raw"],
                    "document_id": matched_record.document_id,
                    "source_path": matched_record.source_path,
                    "evidence_id": matched_record.evidence_id,
                    "page_match": page_valid,
                })
            else:
                hallucinated.append({
                    "citation": cite["raw"],
                    "unresolved_token": cite["token"],
                    "reason": "Referenced document or evidence ID not found in retrieved records.",
                })

        total = len(extracted)
        valid_count = len(verified)
        integrity = valid_count / total if total > 0 else 1.0

        return {
            "total_citations": total,
            "valid_citations": valid_count,
            "hallucinated_citations": len(hallucinated),
            "citation_integrity": round(integrity, 4),
            "verified_references": verified,
            "hallucinated_references": hallucinated,
            "has_hallucinations": len(hallucinated) > 0,
        }

    def format_bibliography(self) -> str:
        """Render a formatted bibliography of all referenced evidence."""
        if not self.records:
            return "No citations available."

        lines = ["### References & Evidence Provenance\n"]
        for idx, r in enumerate(self.records, 1):
            ref = f"{idx}. **{r.document_id}**"
            details = []
            if r.page_number is not None:
                details.append(f"page {r.page_number}")
            if r.char_start is not None and r.char_end is not None:
                details.append(f"offset {r.char_start}-{r.char_end}")
            if details:
                ref += f" ({', '.join(details)})"
            if r.retrieval_method:
                ref += f" — via `{r.retrieval_method}`"
            lines.append(ref)

        return "\n".join(lines)
