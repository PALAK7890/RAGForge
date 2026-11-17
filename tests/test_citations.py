"""
Unit tests for CitationValidator, citation parsing, provenance tracking, and hallucination detection.
"""

import pytest

from knowledge.agentic.citations import CitationValidator
from knowledge.agentic.schemas import EvidenceRecord


@pytest.fixture
def sample_evidence():
    return [
        EvidenceRecord(
            evidence_id="ev_001",
            chunk_id="chk_1",
            document_id="annual_report_2024.pdf",
            source_path="/docs/annual_report_2024.pdf",
            page_number=3,
            char_start=500,
            char_end=750,
            text="Operating profit increased by 14.2% reaching $1.1B.",
            retrieval_method="search",
        ),
        EvidenceRecord(
            evidence_id="ev_002",
            chunk_id="chk_2",
            document_id="cybersecurity_policy.docx",
            source_path="/docs/cybersecurity_policy.docx",
            page_number=1,
            char_start=0,
            char_end=300,
            text="Passwords must be rotated every 90 days across all systems.",
            retrieval_method="open",
        ),
    ]


def test_extract_citations_various_formats(sample_evidence):
    validator = CitationValidator(sample_evidence)
    text = (
        "Operating profit rose [annual_report_2024.pdf, p. 3]. "
        "Also passwords require rotation [cybersecurity_policy.docx, p. 1, chars 0-300]. "
        "Check this link [Download Guide](http://example.com/guide.pdf)."
    )
    extracted = validator.extract_citations(text)
    # The markdown link should be ignored, only the 2 document citations parsed
    assert len(extracted) == 2
    assert extracted[0]["token"] == "annual_report_2024.pdf"
    assert extracted[0]["page"] == 3

    assert extracted[1]["token"] == "cybersecurity_policy.docx"
    assert extracted[1]["page"] == 1
    assert extracted[1]["char_start"] == 0
    assert extracted[1]["char_end"] == 300


def test_validate_all_valid_citations(sample_evidence):
    validator = CitationValidator(sample_evidence)
    answer = (
        "The operating profit was $1.1B [annual_report_2024.pdf, p. 3]. "
        "Security policies mandate rotation [cybersecurity_policy.docx]."
    )
    report = validator.validate(answer)
    assert report["total_citations"] == 2
    assert report["valid_citations"] == 2
    assert report["hallucinated_citations"] == 0
    assert report["citation_integrity"] == 1.0
    assert report["has_hallucinations"] is False


def test_detect_hallucinated_citations(sample_evidence):
    validator = CitationValidator(sample_evidence)
    answer = (
        "Profit grew to $1.1B [annual_report_2024.pdf]. "
        "However, insider trading was noted in [confidential_memo_2024.txt] "
        "and confirmed by [fake_audit_report.pdf, p. 99]."
    )
    report = validator.validate(answer)
    assert report["total_citations"] == 3
    assert report["valid_citations"] == 1
    assert report["hallucinated_citations"] == 2
    assert report["citation_integrity"] == round(1 / 3, 4)
    assert report["has_hallucinations"] is True

    hallucinated_tokens = [h["unresolved_token"] for h in report["hallucinated_references"]]
    assert "confidential_memo_2024.txt" in hallucinated_tokens
    assert "fake_audit_report.pdf" in hallucinated_tokens


def test_format_bibliography(sample_evidence):
    validator = CitationValidator(sample_evidence)
    bib = validator.format_bibliography()
    assert "### References & Evidence Provenance" in bib
    assert "annual_report_2024.pdf" in bib
    assert "page 3" in bib
    assert "cybersecurity_policy.docx" in bib
