"""
app/ai_engine/retrieval/chunking.py
Deterministic, boundary-aware text chunking for evidence extraction.
Decomposes large documents into coherent, context-rich chunks while strictly
respecting character limits and sliding-window overlap.
"""
from dataclasses import dataclass
import re
from typing import List


@dataclass
class TextChunk:
    """Represents a discrete text slice with positional metadata."""
    text: str
    chunk_index: int
    total_chunks: int
    char_count: int
    start_char: int
    end_char: int


class TextChunker:
    """
    Deterministic text chunking engine.
    Splits content along natural boundaries (paragraphs -> sentences -> words)
    maintaining context while enforcing character budgets.
    """

    def __init__(self, max_chunk_chars: int = 1000, overlap_chars: int = 100):
        if max_chunk_chars < 50:
            raise ValueError("max_chunk_chars must be at least 50.")
        if overlap_chars < 0:
            raise ValueError("overlap_chars cannot be negative.")
        if overlap_chars >= max_chunk_chars:
            raise ValueError("overlap_chars must be less than max_chunk_chars.")

        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars

    def chunk_text(self, text: str) -> List[TextChunk]:
        """
        Splits input text into a list of TextChunk objects.
        Returns empty list if input text is empty or purely whitespace.
        """
        if not text or not isinstance(text, str):
            return []

        cleaned = text.strip()
        if not cleaned:
            return []

        # If text fits cleanly within a single chunk, return immediately
        if len(cleaned) <= self.max_chunk_chars:
            return [
                TextChunk(
                    text=cleaned,
                    chunk_index=0,
                    total_chunks=1,
                    char_count=len(cleaned),
                    start_char=0,
                    end_char=len(cleaned),
                )
            ]

        # Break text into atomic semantic units (paragraphs or sentences)
        units = self._split_into_units(cleaned)

        raw_chunks: List[str] = []
        current_units: List[str] = []
        current_len = 0

        for unit in units:
            unit_len = len(unit)

            # If unit itself is oversized (e.g. giant unformatted blob), hard-slice it
            if unit_len > self.max_chunk_chars:
                # Flush existing buffer first
                if current_units:
                    chunk_str = "".join(current_units).strip()
                    if chunk_str:
                        raw_chunks.append(chunk_str)
                    current_units = []
                    current_len = 0

                # Slice oversized unit directly
                slices = self._hard_slice_oversized(unit)
                raw_chunks.extend(slices)
                continue

            # Check if adding unit exceeds max_chunk_chars
            if current_len + unit_len > self.max_chunk_chars and current_units:
                chunk_str = "".join(current_units).strip()
                if chunk_str:
                    raw_chunks.append(chunk_str)

                # Prepare overlap from previous chunk
                overlap_prefix = self._extract_overlap(chunk_str)
                current_units = [overlap_prefix] if overlap_prefix else []
                current_len = len(overlap_prefix) if overlap_prefix else 0

            current_units.append(unit)
            current_len += unit_len

        # Flush any remaining units
        if current_units:
            chunk_str = "".join(current_units).strip()
            if chunk_str:
                raw_chunks.append(chunk_str)

        # Deduplicate consecutive identical chunks if any overlap artifact occurred
        filtered_chunks: List[str] = []
        for c in raw_chunks:
            c_clean = c.strip()
            if c_clean and (not filtered_chunks or filtered_chunks[-1] != c_clean):
                filtered_chunks.append(c_clean)

        total = len(filtered_chunks)
        results: List[TextChunk] = []
        current_pos = 0

        for idx, c_text in enumerate(filtered_chunks):
            start = cleaned.find(c_text[:30], current_pos) if len(c_text) >= 30 else cleaned.find(c_text, current_pos)
            if start == -1:
                start = current_pos
            end = start + len(c_text)
            current_pos = max(start, current_pos)

            results.append(
                TextChunk(
                    text=c_text,
                    chunk_index=idx,
                    total_chunks=total,
                    char_count=len(c_text),
                    start_char=start,
                    end_char=end,
                )
            )

        return results

    def _split_into_units(self, text: str) -> List[str]:
        """
        Splits text by double newlines (paragraphs), then by sentence endings,
        preserving delimiter spaces.
        """
        # Split on paragraph boundaries first
        paragraphs = re.split(r"(\n\n+)", text)
        units: List[str] = []

        for p in paragraphs:
            if not p:
                continue
            if re.match(r"^\n\n+$", p):
                units.append(p)
                continue

            # Within paragraph, split into sentences (e.g. ending in '.', '!', '?')
            sentences = re.split(r"((?<=[.!?])\s+)", p)
            for s in sentences:
                if s:
                    units.append(s)

        return units

    def _extract_overlap(self, text: str) -> str:
        """
        Extracts up to self.overlap_chars from the end of the text,
        attempting to break at word boundaries.
        """
        if self.overlap_chars <= 0 or not text:
            return ""

        candidate = text[-self.overlap_chars:]
        # Find first space to avoid partial word cutoffs
        space_idx = candidate.find(" ")
        if space_idx != -1 and space_idx < len(candidate) - 10:
            candidate = candidate[space_idx + 1:]

        return candidate.strip() + " " if candidate.strip() else ""

    def _hard_slice_oversized(self, text: str) -> List[str]:
        """
        Hard slices oversized text into chunks <= max_chunk_chars,
        favoring whitespace breaks when possible.
        """
        slices: List[str] = []
        remaining = text.strip()

        while len(remaining) > self.max_chunk_chars:
            cutoff = self.max_chunk_chars
            # Attempt to split at last whitespace within cutoff
            last_space = remaining[:cutoff].rfind(" ")
            if last_space > self.max_chunk_chars // 2:
                cutoff = last_space

            part = remaining[:cutoff].strip()
            if part:
                slices.append(part)

            remaining = remaining[cutoff:].strip()

        if remaining:
            slices.append(remaining)

        return slices
