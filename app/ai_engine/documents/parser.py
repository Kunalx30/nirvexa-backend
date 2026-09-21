"""
app/ai_engine/documents/parser.py
Parser for user documents supporting PDF, TXT, and Markdown formats.
Uses pre-installed pdfplumber for PDF extraction and standard library text decoders.
"""
import io
import os
import re
import logging
from typing import Dict, Any, Tuple
from app.ai_engine.documents.schemas import ParsedDocument

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


class DocumentParserError(ValueError):
    """Raised when document parsing fails due to corrupt, invalid, or unsupported format."""
    pass


class DocumentParser:
    """
    Extracts text and metadata from supported document formats.
    """

    @classmethod
    def get_extension(cls, filename: str) -> str:
        _, ext = os.path.splitext(filename.lower())
        return ext

    @classmethod
    def is_supported(cls, filename: str) -> bool:
        return cls.get_extension(filename) in SUPPORTED_EXTENSIONS

    @classmethod
    def sanitize_text(cls, text: str) -> str:
        if not text:
            return ""
        # Remove null bytes and carriage-return artifacts
        cleaned = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
        # Collapse excessive whitespace
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    @classmethod
    def parse(cls, file_bytes: bytes, filename: str) -> ParsedDocument:
        """
        Parses raw bytes of a file and returns a structured ParsedDocument.
        """
        if not file_bytes:
            raise DocumentParserError("Uploaded file is empty.")

        ext = cls.get_extension(filename)
        if ext not in SUPPORTED_EXTENSIONS:
            raise DocumentParserError(
                f"Unsupported file format '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        title = os.path.splitext(os.path.basename(filename))[0]

        if ext == ".pdf":
            return cls._parse_pdf(file_bytes, filename, title)
        elif ext in (".txt", ".md"):
            return cls._parse_text(file_bytes, filename, title, ext[1:])

        raise DocumentParserError(f"Unsupported format: {ext}")

    @classmethod
    def _parse_text(cls, file_bytes: bytes, filename: str, title: str, file_type: str) -> ParsedDocument:
        for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                decoded = file_bytes.decode(encoding)
                cleaned = cls.sanitize_text(decoded)
                if not cleaned:
                    raise DocumentParserError("Text document contains no readable content.")
                return ParsedDocument(
                    text=cleaned,
                    title=title,
                    file_type=file_type,
                    char_count=len(cleaned),
                    metadata={"filename": filename, "encoding": encoding},
                )
            except UnicodeDecodeError:
                continue

        raise DocumentParserError("Failed to decode text file with supported encodings.")

    @classmethod
    def _parse_pdf(cls, file_bytes: bytes, filename: str, title: str) -> ParsedDocument:
        # Magic bytes check
        if not file_bytes.startswith(b"%PDF-"):
            raise DocumentParserError("Invalid PDF header: file does not appear to be a valid PDF.")

        try:
            import pdfplumber
        except ImportError as exc:
            logger.error("[DocumentParser] pdfplumber is not installed: %s", exc)
            raise DocumentParserError("PDF parser is not available.") from exc

        page_texts = []
        metadata: Dict[str, Any] = {"filename": filename}

        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                metadata["total_pages"] = len(pdf.pages)
                for idx, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    clean_page = cls.sanitize_text(page_text)
                    if clean_page:
                        page_texts.append(clean_page)

        except Exception as exc:
            logger.error("[DocumentParser] PDF extraction error for %s: %s", filename, exc)
            raise DocumentParserError(f"Failed to extract content from PDF: {str(exc)}") from exc

        full_text = "\n\n".join(page_texts).strip()
        if not full_text:
            raise DocumentParserError("PDF contains no extractable text (it may be scanned images or password protected).")

        return ParsedDocument(
            text=full_text,
            title=title,
            file_type="pdf",
            char_count=len(full_text),
            metadata=metadata,
        )
