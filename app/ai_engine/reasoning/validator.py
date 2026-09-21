"""
app/ai_engine/reasoning/validator.py
Citation and Grounding Validator.
Analyzes LLM-generated text, extracts source citations, validates them against
the provided EvidencePack, and flags hallucinated or unsupported claims.
"""
import re
from typing import Dict, List, Tuple, Set

from app.ai_engine.reasoning.schemas import CitationItem, GroundingStatus
from app.ai_engine.retrieval.schemas import EvidenceItem


class CitationValidator:
    """
    Validates citation markers in LLM-generated output against known evidence sources.
    """

    # Matches bracketed citation markers: [S1], [S2], [S10], etc.
    CITATION_REGEX = re.compile(r"\[S(\d+)\]")

    @classmethod
    def validate(
        cls,
        generated_text: str,
        citation_map: Dict[str, EvidenceItem],
        total_evidence_count: int,
    ) -> Tuple[GroundingStatus, List[CitationItem], List[str]]:
        """
        Validates citations in generated text.
        Returns:
            (grounding_status, valid_citations, hallucinated_citation_ids)
        """
        if not generated_text:
            return "insufficient_evidence", [], []

        # Find all cited markers in order of appearance
        matches = cls.CITATION_REGEX.findall(generated_text)
        cited_keys = [f"S{m}" for m in matches]

        valid_citations: List[CitationItem] = []
        hallucinated_ids: List[str] = []
        seen_valid: Set[str] = set()

        for ckey in cited_keys:
            if ckey in citation_map:
                if ckey not in seen_valid:
                    seen_valid.add(ckey)
                    ev_item = citation_map[ckey]
                    valid_citations.append(
                        CitationItem(
                            citation_id=ckey,
                            source_id=ev_item.source_id,
                            title=ev_item.title,
                            url=ev_item.url,
                            source_type=ev_item.source_type,
                            snippet=ev_item.text[:200] if ev_item.text else None,
                            metadata=dict(ev_item.metadata),
                        )
                    )
            else:
                if ckey not in hallucinated_ids:
                    hallucinated_ids.append(ckey)

        # Determine GroundingStatus
        text_lower = generated_text.lower()
        is_insufficient = (
            "insufficient evidence" in text_lower
            or "insufficient information" in text_lower
            or "no relevant evidence" in text_lower
            or "unable to find" in text_lower
        )

        if total_evidence_count == 0 or (is_insufficient and not valid_citations):
            status: GroundingStatus = "insufficient_evidence"
        elif hallucinated_ids:
            # Contains invalid citations referencing nonexistent sources
            status = "partially_grounded"
        elif valid_citations:
            # All citations are verified and valid
            status = "grounded"
        else:
            # Zero citations found in generated text
            status = "insufficient_evidence" if is_insufficient else "unsupported"

        return status, valid_citations, hallucinated_ids
