import logging
import feedparser
import requests
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime

logger = logging.getLogger(__name__)

# ── RSS Sources ───────────────────────────────────────────────────────────────
# category is fixed per source — clean and predictable
RSS_SOURCES = [
    {
        'name':     'TechCrunch',
        'url':      'https://techcrunch.com/feed/',
        'category': 'ai',
    },
    {
        'name':     'YourStory',
        'url':      'https://yourstory.com/feed',
        'category': 'startups',
    },
    {
        'name':     'Economic Times Tech',
        'url':      'https://economictimes.indiatimes.com/tech/rss',
        'category': 'general',
    },
    {
        'name':     'Hacker News',
        'url':      'https://news.ycombinator.com/rss',
        'category': 'general',
    },
    {
        'name':     'Inc42',
        'url':      'https://inc42.com/feed/',
        'category': 'startups',
    },
]

# ── Category keyword override ─────────────────────────────────────────────────
# Even if source has a fixed category, we scan the title to upgrade it
# e.g. a TechCrunch article about "data science jobs" → data-science
KEYWORD_CATEGORY_MAP = {
    'data-science': ['data science', 'machine learning', 'deep learning', 'nlp',
                     'neural network', 'llm', 'large language model', 'pytorch', 'tensorflow'],
    'ai':           ['artificial intelligence', 'ai ', ' ai,', 'openai', 'anthropic',
                     'chatgpt', 'gemini', 'llama', 'generative ai', 'foundation model'],
    'startups':     ['startup', 'funding', 'series a', 'series b', 'seed round',
                     'unicorn', 'venture', 'vc ', 'founder'],
    'govt-jobs':    ['upsc', 'ssc', 'government job', 'psu', 'ncs', 'recruitment',
                     'vacancy', 'admit card', 'result 2025', 'result 2026'],
}


def _detect_category(title: str, default_category: str) -> str:
    """Scan title for keywords to assign best category. Falls back to source default."""
    title_lower = title.lower()
    for category, keywords in KEYWORD_CATEGORY_MAP.items():
        if any(kw in title_lower for kw in keywords):
            return category
    return default_category


def _parse_published_at(entry) -> datetime | None:
    """Safely parse RSS published date."""
    try:
        if hasattr(entry, 'published'):
            return parsedate_to_datetime(entry.published).replace(tzinfo=timezone.utc)
    except Exception:
        pass
    return datetime.now(timezone.utc)


def _summarize_with_mistral(title: str, snippet: str) -> str | None:
    """
    Call Mistral 7B via llm_router helper to generate a 2-line summary.
    Falls back to None if it fails — article still gets saved without summary.
    """
    try:
        from app.services.llm_router import call_mistral_simple

        prompt = (
            f"Article title: {title}\n"
            f"Snippet: {snippet[:300]}\n\n"
            "Write a 2-sentence summary of this article for a tech-savvy Indian job seeker. "
            "Be concise. No bullet points. No preamble."
        )
        return call_mistral_simple(prompt, max_tokens=150)
    except Exception as e:
        logger.warning("[NewsService] Mistral summarization failed: %s", e)
        return None


def fetch_and_cache_news() -> dict:
    """
    Main function called by scheduler every 4 hours.
    Fetches all RSS feeds, deduplicates against DB, summarizes new articles, saves to DB.
    Returns stats dict.
    """
    from app.models.news_cache import NewsCache
    from app.database.db import db

    stats = {'fetched': 0, 'inserted': 0, 'skipped': 0, 'errors': 0}

    for source in RSS_SOURCES:
        try:
            feed = feedparser.parse(source['url'])
            entries = feed.entries[:15]  # cap at 15 per source per run
            logger.info("[News] %s — %d entries fetched", source['name'], len(entries))

            for entry in entries:
                stats['fetched'] += 1

                url   = getattr(entry, 'link', None)
                title = getattr(entry, 'title', None)

                if not url or not title:
                    stats['skipped'] += 1
                    continue

                # Dedup check — url is unique in DB
                existing = NewsCache.query.filter_by(url=url).first()
                if existing:
                    stats['skipped'] += 1
                    continue

                # Category detection
                category = _detect_category(title, source['category'])

                # Published date
                published_at = _parse_published_at(entry)

                # Skip articles older than 7 days — keeps feed fresh
                if published_at and (datetime.now(timezone.utc) - published_at) > timedelta(days=7):
                    stats['skipped'] += 1
                    continue

                # Snippet for summarization
                snippet = ''
                if hasattr(entry, 'summary'):
                    snippet = entry.summary

                # AI summary via Mistral 7B
                ai_summary = _summarize_with_mistral(title, snippet)

                # Save to DB
                article = NewsCache(
                    title=title[:500],
                    source=source['name'],
                    url=url[:1000],
                    published_at=published_at,
                    category=category,
                    ai_summary=ai_summary,
                )
                db.session.add(article)

                try:
                    db.session.commit()
                    stats['inserted'] += 1
                except Exception as e:
                    db.session.rollback()
                    logger.error("[News] DB insert failed for '%s': %s", title, e)
                    stats['errors'] += 1

        except Exception as e:
            logger.error("[News] Source '%s' failed: %s", source['name'], e)
            stats['errors'] += 1

    logger.info(
        "[News] Run complete — fetched=%d inserted=%d skipped=%d errors=%d",
        stats['fetched'], stats['inserted'], stats['skipped'], stats['errors']
    )
    return stats