"""
app/ai_engine/extraction/content_extractor.py
Robust HTML text cleaner and metadata extractor using standard library html.parser.
Preserves full clean text for downstream RAG chunking without premature truncation.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
import re
from typing import Optional
import urllib.parse

from app.ai_engine.schemas.research import SourceMetadata


@dataclass
class ExtractedContent:
    """Extracted text and metadata from a webpage."""
    title: str
    text: str
    metadata: SourceMetadata


class _HTMLContentParser(HTMLParser):
    """HTML Parser stripping non-content tags and extracting text + metadata."""

    # Tags whose inner content must be discarded entirely
    IGNORED_TAGS = {
        "script",
        "style",
        "noscript",
        "nav",
        "footer",
        "header",
        "aside",
        "svg",
        "iframe",
        "form",
        "button",
        "select",
        "textarea",
    }

    # Tags that introduce block separation
    BLOCK_TAGS = {
        "p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "article", "section", "blockquote", "pre", "tr", "br"
    }

    def __init__(self):
        super().__init__()
        self.ignore_depth = 0
        self.text_parts = []
        self.current_tag = ""
        self.in_title = False
        self.title_parts = []

        # Metadata attributes
        self.meta_description: Optional[str] = None
        self.meta_author: Optional[str] = None
        self.meta_published_at: Optional[str] = None
        self.canonical_url: Optional[str] = None
        self.og_title: Optional[str] = None

    def handle_starttag(self, tag, attrs):
        tag_lower = tag.lower()
        self.current_tag = tag_lower
        attr_dict = {k.lower(): v for k, v in attrs if v is not None}

        if tag_lower in self.IGNORED_TAGS:
            self.ignore_depth += 1
            return

        if tag_lower == "title":
            self.in_title = True
            return

        if tag_lower == "meta":
            name = attr_dict.get("name", "").lower()
            prop = attr_dict.get("property", "").lower()
            content = attr_dict.get("content", "").strip()

            if content:
                # Description
                if (name == "description" or prop == "og:description") and not self.meta_description:
                    self.meta_description = content

                # Title
                if (prop in ("og:title", "twitter:title")) and not self.og_title:
                    self.og_title = content

                # Author
                if (name in ("author", "byl", "article:author") or prop == "article:author") and not self.meta_author:
                    self.meta_author = content

                # Published date
                if (name in ("date", "pubdate", "article:published_time") or prop == "article:published_time") and not self.meta_published_at:
                    self.meta_published_at = content

        elif tag_lower == "link":
            rel = attr_dict.get("rel", "").lower()
            href = attr_dict.get("href", "").strip()
            if rel == "canonical" and href:
                self.canonical_url = href

        elif tag_lower in self.BLOCK_TAGS:
            self.text_parts.append("\n")

    def handle_endtag(self, tag):
        tag_lower = tag.lower()
        if tag_lower in self.IGNORED_TAGS:
            if self.ignore_depth > 0:
                self.ignore_depth -= 1
            return

        if tag_lower == "title":
            self.in_title = False
            return

        if tag_lower in self.BLOCK_TAGS:
            self.text_parts.append("\n")

    def handle_data(self, data):
        if self.ignore_depth > 0:
            return

        if self.in_title:
            self.title_parts.append(data)
            return

        stripped = data.strip()
        if stripped:
            self.text_parts.append(data)


class ContentExtractor:
    """Extracts clean text and metadata from raw HTML."""

    @classmethod
    def extract(cls, html: str, url: str) -> ExtractedContent:
        if not html:
            domain = urllib.parse.urlparse(url).netloc
            return ExtractedContent(
                title="",
                text="",
                metadata=SourceMetadata(domain=domain, retrieved_at=datetime.now(timezone.utc).isoformat()),
            )

        parser = _HTMLContentParser()
        try:
            parser.feed(html)
        except Exception:
            # Tolerant parsing fallback
            pass

        # 1. Resolve title
        title = "".join(parser.title_parts).strip()
        if not title and parser.og_title:
            title = parser.og_title.strip()

        # 2. Normalize and clean body text
        raw_text = "".join(parser.text_parts)
        cleaned_text = cls._normalize_whitespace(raw_text)

        # 3. Resolve domain
        domain = urllib.parse.urlparse(url).netloc

        metadata = SourceMetadata(
            title=title,
            description=parser.meta_description,
            author=parser.meta_author,
            published_at=parser.meta_published_at,
            canonical_url=parser.canonical_url,
            domain=domain,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
        )

        return ExtractedContent(
            title=title,
            text=cleaned_text,
            metadata=metadata,
        )

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Collapse multiple spaces and consecutive newlines into clean readable text."""
        # Replace non-breaking spaces
        text = text.replace("\xa0", " ").replace("&nbsp;", " ")
        # Replace multiple spaces/tabs with single space
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse 3 or more newlines into double newline
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        # Strip leading and trailing whitespace per line
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(lines).strip()
