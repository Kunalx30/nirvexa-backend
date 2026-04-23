import requests
import time
import logging
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def normalize_job(raw: dict) -> dict:
    """Standardize all fields across sources."""
    skills = raw.get('skills') or []
    if isinstance(skills, str):
        skills = [s.strip().lower() for s in skills.split(',') if s.strip()]
    else:
        skills = [s.strip().lower() for s in skills if s.strip()]

    salary = raw.get('salary') or ''
    salary = salary.replace('₹', '').replace('$', '').strip()

    return {
        'title':       (raw.get('title') or '').strip(),
        'company':     (raw.get('company') or '').strip(),
        'location':    (raw.get('location') or 'Remote').strip(),
        'skills':      skills,
        'salary':      salary,
        'source':      raw.get('source', 'unknown'),
        'job_type':    raw.get('job_type', 'fulltime'),
        'apply_url':   raw.get('apply_url') or '',
        'description': raw.get('description') or '',
        'posted_at':   raw.get('posted_at') or datetime.now(timezone.utc),
        'expires_at':  datetime.now(timezone.utc) + timedelta(days=30),
    }


def scrape_remotive() -> list[dict]:
    """
    Remotive.com — free open REST API, no auth required.
    Returns remote tech jobs globally.
    """
    url = "https://remotive.com/api/remote-jobs"
    try:
        resp = requests.get(url, timeout=15, headers={
            'User-Agent': 'NirVexa/1.0 Job Aggregator'
        })
        resp.raise_for_status()
        data = resp.json()
        items = data.get('jobs', [])

        jobs = []
        for item in items:
            # Parse posted date
            posted_at = datetime.now(timezone.utc)
            try:
                posted_at = datetime.strptime(
                    item.get('publication_date', ''), "%Y-%m-%dT%H:%M:%S"
                )
            except Exception:
                pass

            raw = {
                'title':       item.get('title', ''),
                'company':     item.get('company_name', ''),
                'location':    item.get('candidate_required_location') or 'Remote',
                'skills':      item.get('tags') or [],
                'salary':      item.get('salary') or '',
                'source':      'remotive',
                'job_type':    'remote',
                'apply_url':   item.get('url', ''),
                'description': item.get('description', '')[:2000],  # trim long HTML
                'posted_at':   posted_at,
            }
            jobs.append(normalize_job(raw))

        logger.info(f"[Remotive] Fetched {len(jobs)} jobs")
        return jobs

    except Exception as e:
        logger.error(f"[Remotive] Scraper failed: {e}")
        return []


def scrape_github_jobs() -> list[dict]:
    """
    Arbeitnow — free open API, no auth. 
    Good mix of remote + EU tech jobs.
    """
    url = "https://www.arbeitnow.com/api/job-board-api"
    try:
        resp = requests.get(url, timeout=15, headers={
            'User-Agent': 'NirVexa/1.0 Job Aggregator'
        })
        resp.raise_for_status()
        data = resp.json()
        items = data.get('data', [])

        jobs = []
        for item in items:
            posted_at = datetime.now(timezone.utc)
            try:
                posted_at = datetime.fromtimestamp(item.get('created_at', 0), tz=timezone.utc)
            except Exception:
                pass

            raw = {
                'title':       item.get('title', ''),
                'company':     item.get('company_name', ''),
                'location':    item.get('location') or 'Remote',
                'skills':      item.get('tags') or [],
                'salary':      '',
                'source':      'arbeitnow',
                'job_type':    'remote' if item.get('remote') else 'fulltime',
                'apply_url':   item.get('url', ''),
                'description': (item.get('description') or '')[:2000],
                'posted_at':   posted_at,
            }
            jobs.append(normalize_job(raw))

        logger.info(f"[Arbeitnow] Fetched {len(jobs)} jobs")
        return jobs

    except Exception as e:
        logger.error(f"[Arbeitnow] Scraper failed: {e}")
        return []

    
def scrape_internshala() -> list[dict]:
    """
    Internshala — BeautifulSoup scraper.
    Best source for Indian fresher + internship roles.
    """
    base_url = "https://internshala.com"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    jobs = []

    # Scrape both internships and fresher jobs
    endpoints = [
        ("/internships", "internship"),
        ("/jobs/fresher-jobs", "fulltime"),
    ]

    for path, job_type in endpoints:
        try:
            resp = requests.get(base_url + path, headers=headers, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'lxml')

            # Each listing card
            cards = soup.select('.individual_internship')
            logger.info(f"[Internshala] {path} → found {len(cards)} cards")

            for card in cards:
                try:
                    title = card.select_one('.job-internship-name, .profile')
                    company = card.select_one('.company-name')
                    location = card.select_one('.location_link, .locations')
                    stipend = card.select_one('.stipend, .salary')
                    link = card.select_one('a.view_detail_button, a[href*="/internship/detail"]')

                    apply_url = ''
                    if link and link.get('href'):
                        href = link['href']
                        apply_url = href if href.startswith('http') else base_url + href

                    # Skills from tags
                    skill_tags = card.select('.round_tabs span, .skills span')
                    skills = [s.get_text(strip=True) for s in skill_tags if s.get_text(strip=True)]

                    raw = {
                        'title':    title.get_text(strip=True) if title else '',
                        'company':  company.get_text(strip=True) if company else '',
                        'location': location.get_text(strip=True) if location else 'India',
                        'skills':   skills,
                        'salary':   stipend.get_text(strip=True) if stipend else '',
                        'source':   'internshala',
                        'job_type': job_type,
                        'apply_url': apply_url,
                        'description': '',
                        'posted_at': datetime.now(timezone.utc),
                    }

                    if raw['title'] and raw['company']:
                        jobs.append(normalize_job(raw))

                except Exception as e:
                    logger.warning(f"[Internshala] Card parse error: {e}")
                    continue

            time.sleep(1)  # be polite between requests

        except Exception as e:
            logger.error(f"[Internshala] Failed on {path}: {e}")
            continue

    logger.info(f"[Internshala] Total fetched: {len(jobs)}")
    return jobs


def scrape_greenhouse_companies() -> list[dict]:
    """
    Greenhouse ATS — free public API, no auth.
    Pulls directly from Indian startup career pages.
    """
    # These companies use Greenhouse and have India-relevant hiring
    companies = [
       ('postman', 'Postman'),
        ('naukri', 'Naukri'), 
    ]

    jobs = []
    for slug, company_name in companies:
        try:
            url = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true"
            resp = requests.get(url, timeout=15, headers={
                'User-Agent': 'NirVexa/1.0 Job Aggregator'
            })
            resp.raise_for_status()
            data = resp.json()
            items = data.get('jobs', [])

            for item in items:
                # Extract location
                offices = item.get('offices') or []
                location = ', '.join([o.get('name', '') for o in offices]) or 'India'

                # Extract department as skill tag
                depts = item.get('departments') or []
                skills = [d.get('name', '') for d in depts if d.get('name')]

                raw = {
                    'title':       item.get('title', ''),
                    'company':     company_name,
                    'location':    location,
                    'skills':      skills,
                    'salary':      '',
                    'source':      'greenhouse',
                    'job_type':    'fulltime',
                    'apply_url':   item.get('absolute_url', ''),
                    'description': (item.get('content') or '')[:2000],
                    'posted_at':   datetime.utcnow(),
                }

                if raw['title']:
                    jobs.append(normalize_job(raw))

            logger.info(f"[Greenhouse] {company_name}: {len(items)} jobs")
            time.sleep(0.5)

        except Exception as e:
            logger.error(f"[Greenhouse] {company_name} failed: {e}")
            continue

    logger.info(f"[Greenhouse] Total: {len(jobs)} jobs")
    return jobs


def scrape_lever_companies() -> list[dict]:
    """
    Lever ATS — free public API, no auth.
    Pulls directly from Indian startup career pages.
    """
    companies = [
        ('cred', 'CRED'),
    ]

    jobs = []
    for slug, company_name in companies:
        try:
            url = f"https://api.lever.co/v0/postings/{slug}?mode=json"
            resp = requests.get(url, timeout=15, headers={
                'User-Agent': 'NirVexa/1.0 Job Aggregator'
            })
            resp.raise_for_status()
            items = resp.json()

            if not isinstance(items, list):
                logger.warning(f"[Lever] {company_name}: unexpected response format")
                continue

            for item in items:
                categories = item.get('categories', {})
                location = categories.get('location') or categories.get('allLocations') or 'India'
                if isinstance(location, list):
                    location = ', '.join(location)

                team = categories.get('team', '')
                commitment = categories.get('commitment', 'fulltime').lower()

                job_type = 'fulltime'
                if 'intern' in commitment:
                    job_type = 'internship'
                elif 'contract' in commitment:
                    job_type = 'fulltime'

                raw = {
                    'title':       item.get('text', ''),
                    'company':     company_name,
                    'location':    location,
                    'skills':      [team] if team else [],
                    'salary':      '',
                    'source':      'lever',
                    'job_type':    job_type,
                    'apply_url':   item.get('hostedUrl', ''),
                    'description': (item.get('descriptionPlain') or '')[:2000],
                    'posted_at':   datetime.utcfromtimestamp(
                                       item.get('createdAt', 0) / 1000
                                   ) if item.get('createdAt') else datetime.utcnow(),
                }

                if raw['title']:
                    jobs.append(normalize_job(raw))

            logger.info(f"[Lever] {company_name}: {len(items)} jobs")
            time.sleep(0.5)

        except Exception as e:
            logger.error(f"[Lever] {company_name} failed: {e}")
            continue

    logger.info(f"[Lever] Total: {len(jobs)} jobs")
    return jobs

def scrape_indian_startups_greenhouse() -> list[dict]:
    """
    Expanded Greenhouse scraper — Indian startups + MNCs with India offices.
    Covers Hyderabad, Bangalore, Pune, Delhi, Gurgaon, Chennai, Noida.
    """
    companies = [
         ('postman', 'Postman'),
        ('naukri', 'Naukri'),
        ('slintel', 'Slintel'),
    ]

    jobs = []
    for slug, company_name in companies:
        try:
            url = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true"
            resp = requests.get(url, timeout=10, headers={
                'User-Agent': 'NirVexa/1.0 Job Aggregator'
            })
            if resp.status_code != 200:
                logger.warning(f"[GH-India] {company_name}: HTTP {resp.status_code} — skipping")
                time.sleep(0.3)
                continue

            data = resp.json()
            items = data.get('jobs', [])

            for item in items:
                offices = item.get('offices') or []
                location = ', '.join([o.get('name', '') for o in offices]) or 'India'

                depts = item.get('departments') or []
                skills = [d.get('name', '') for d in depts if d.get('name')]

                raw = {
                    'title':       item.get('title', ''),
                    'company':     company_name,
                    'location':    location,
                    'skills':      skills,
                    'salary':      '',
                    'source':      'greenhouse',
                    'job_type':    'fulltime',
                    'apply_url':   item.get('absolute_url', ''),
                    'description': (item.get('content') or '')[:2000],
                    'posted_at':   datetime.utcnow(),
                }
                if raw['title']:
                    jobs.append(normalize_job(raw))

            logger.info(f"[GH-India] {company_name}: {len(items)} jobs")
            time.sleep(0.4)

        except Exception as e:
            logger.error(f"[GH-India] {company_name} failed: {e}")
            continue

    logger.info(f"[GH-India] Total: {len(jobs)}")
    return jobs


def scrape_indian_startups_lever() -> list[dict]:
    """
    Lever scraper — VERIFIED slugs only for Indian companies.
    All slugs confirmed working via jobs.lever.co.
    """
    companies = [
        # Confirmed working slugs
        ('meesho', 'Meesho'),           # Bangalore — ecommerce
        ('paytm', 'Paytm'),             # Noida — fintech, huge hiring

    ]

    jobs = []
    for slug, company_name in companies:
        try:
            url = f"https://api.lever.co/v0/postings/{slug}?mode=json"
            resp = requests.get(url, timeout=10, headers={
                'User-Agent': 'NirVexa/1.0 Job Aggregator'
            })
            if resp.status_code != 200:
                logger.warning(f"[Lever-India] {company_name} ({slug}): {resp.status_code} — skip")
                time.sleep(0.3)
                continue

            items = resp.json()
            if not isinstance(items, list):
                logger.warning(f"[Lever-India] {company_name}: unexpected format")
                continue

            count = 0
            for item in items:
                categories = item.get('categories', {})
                location = categories.get('location') or 'India'
                if isinstance(location, list):
                    location = ', '.join(location)

                team = categories.get('team', '')
                commitment = (categories.get('commitment') or 'fulltime').lower()
                job_type = 'internship' if 'intern' in commitment else 'fulltime'

                raw = {
                    'title':       item.get('text', ''),
                    'company':     company_name,
                    'location':    location,
                    'skills':      [team] if team else [],
                    'salary':      '',
                    'source':      'lever',
                    'job_type':    job_type,
                    'apply_url':   item.get('hostedUrl', ''),
                    'description': (item.get('descriptionPlain') or '')[:2000],
                    'posted_at':   datetime.utcfromtimestamp(
                                       item.get('createdAt', 0) / 1000
                                   ) if item.get('createdAt') else datetime.utcnow(),
                }
                if raw['title']:
                    jobs.append(normalize_job(raw))
                    count += 1

            logger.info(f"[Lever-India] {company_name}: {count} jobs")
            time.sleep(0.4)

        except Exception as e:
            logger.error(f"[Lever-India] {company_name} failed: {e}")
            continue

    logger.info(f"[Lever-India] Total: {len(jobs)}")
    return jobs


def scrape_jobicy_india() -> list[dict]:
    """
    Jobicy — free public API, posts new remote jobs daily.
    Filter by 'india' region and dev/data categories.
    """
    jobs = []
    categories = ['dev', 'data-science', 'marketing', 'hr', 'engineering']

    for industry in categories:
        try:
            url = f"https://jobicy.com/api/v2/remote-jobs?count=50&geo=india&industry={industry}"
            resp = requests.get(url, timeout=15, headers={
                'User-Agent': 'NirVexa/1.0 Job Aggregator'
            })
            if resp.status_code != 200:
                continue

            data = resp.json()
            items = data.get('jobs', [])

            for item in items:
                posted_at = datetime.now(timezone.utc)
                try:
                    posted_at = datetime.strptime(
                        item.get('pubDate', ''), "%Y-%m-%dT%H:%M:%S"
                    )
                except Exception:
                    pass

                raw = {
                    'title':       item.get('jobTitle', ''),
                    'company':     item.get('companyName', ''),
                    'location':    item.get('jobGeo') or 'Remote',
                    'skills':      [item.get('jobIndustry', '')],
                    'salary':      f"{item.get('annualSalaryMin','')} - {item.get('annualSalaryMax','')} {item.get('salaryCurrency','')}".strip(' -'),
                    'source':      'jobicy',
                    'job_type':    item.get('jobType', 'fulltime').replace('-', ''),
                    'apply_url':   item.get('url', ''),
                    'description': item.get('jobDescription', '')[:2000],
                    'posted_at':   posted_at,
                }
                if raw['title'] and raw['company']:
                    jobs.append(normalize_job(raw))

            time.sleep(0.5)

        except Exception as e:
            logger.error(f"[Jobicy] {industry} failed: {e}")

    logger.info(f"[Jobicy] Total: {len(jobs)}")
    return jobs


def scrape_naukri_rss() -> list[dict]:
    """
    Naukri RSS feeds — fresh jobs posted today.
    Multiple city + role feeds parsed with feedparser.
    """
    import feedparser

    feeds = [
        # Format: (feed_url, city_label)
        ("https://www.naukri.com/rss/fresher-jobs-in-hyderabad-2.rss", "Hyderabad"),
        ("https://www.naukri.com/rss/fresher-jobs-in-bangalore-1.rss", "Bangalore"),
        ("https://www.naukri.com/rss/fresher-jobs-in-pune-34.rss", "Pune"),
        ("https://www.naukri.com/rss/fresher-jobs-in-delhi-25.rss", "Delhi"),
        ("https://www.naukri.com/rss/fresher-jobs-in-chennai-44.rss", "Chennai"),
        ("https://www.naukri.com/rss/fresher-jobs-in-noida-272.rss", "Noida"),
        ("https://www.naukri.com/rss/fresher-jobs-in-gurgaon-910.rss", "Gurgaon"),
        ("https://www.naukri.com/rss/software-jobs-3.rss", "India"),
        ("https://www.naukri.com/rss/it-jobs-2.rss", "India"),
        ("https://www.naukri.com/rss/data-science-jobs.rss", "India"),
    ]

    jobs = []
    for feed_url, city in feeds:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                posted_at = datetime.now(timezone.utc)
                try:
                    import email.utils
                    posted_at = datetime(*email.utils.parsedate(entry.get('published', ''))[:6])
                except Exception:
                    pass

                raw = {
                    'title':       entry.get('title', ''),
                    'company':     entry.get('author', '') or entry.get('dc_creator', ''),
                    'location':    city,
                    'skills':      [],
                    'salary':      '',
                    'source':      'naukri',
                    'job_type':    'fulltime',
                    'apply_url':   entry.get('link', ''),
                    'description': entry.get('summary', '')[:2000],
                    'posted_at':   posted_at,
                }
                if raw['title']:
                    jobs.append(normalize_job(raw))

            logger.info(f"[Naukri RSS] {city}: {len(feed.entries)} entries")
            time.sleep(1)

        except Exception as e:
            logger.error(f"[Naukri RSS] {city} failed: {e}")

    logger.info(f"[Naukri RSS] Total: {len(jobs)}")
    return jobs


def scrape_freshersworld() -> list[dict]:
    """
    Freshersworld — India's top fresher job site.
    Scrapes latest fresher openings across all cities.
    """
    jobs = []
    pages = [
        "https://www.freshersworld.com/jobs/freshers-jobs",
        "https://www.freshersworld.com/jobs/it-software-jobs",
        "https://www.freshersworld.com/jobs/engineering-jobs",
    ]

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'
    }

    for url in pages:
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, 'lxml')
            cards = soup.select('.job-container, .jobs-list li, .job-card')

            for card in cards:
                title = card.select_one('.job-title, h3, .title')
                company = card.select_one('.company-name, .company')
                location = card.select_one('.location, .job-location')
                link = card.select_one('a')

                apply_url = ''
                if link and link.get('href'):
                    href = link['href']
                    apply_url = href if href.startswith('http') else 'https://www.freshersworld.com' + href

                raw = {
                    'title':       title.get_text(strip=True) if title else '',
                    'company':     company.get_text(strip=True) if company else '',
                    'location':    location.get_text(strip=True) if location else 'India',
                    'skills':      [],
                    'salary':      '',
                    'source':      'freshersworld',
                    'job_type':    'fulltime',
                    'apply_url':   apply_url,
                    'description': '',
                    'posted_at':   datetime.utcnow(),
                }
                if raw['title'] and raw['company']:
                    jobs.append(normalize_job(raw))

            time.sleep(1)

        except Exception as e:
            logger.error(f"[Freshersworld] {url} failed: {e}")

    logger.info(f"[Freshersworld] Total: {len(jobs)}")
    return jobs

def scrape_jsearch_india() -> list[dict]:
    """
    JSearch API — aggregates LinkedIn, Indeed, Glassdoor, Naukri.
    Free: 500 calls/month. Each call returns ~10 jobs.
    Run daily with 5 searches = 50 fresh Indian jobs/day.
    """
    import os
    api_key = os.getenv('RAPIDAPI_KEY')
    if not api_key:
        logger.error("[JSearch] No RAPIDAPI_KEY in .env")
        return []

    queries = [
        "software engineer jobs in Bangalore India",
        "fresher jobs in Hyderabad India",
        "data science jobs in Pune India",
        "python developer jobs in Delhi India",
        "frontend developer jobs in Chennai India",
    ]

    jobs = []
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }

    for query in queries:
        try:
            url = "https://jsearch.p.rapidapi.com/search"
            params = {
                "query": query,
                "page": "1",
                "num_pages": "1",
                "date_posted": "today",  # ONLY TODAY'S JOBS
                "country": "in",
            }
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"[JSearch] {query}: {resp.status_code}")
                continue

            data = resp.json()
            items = data.get('data', [])

            for item in items:
                posted_at = datetime.now(timezone.utc)
                ts = item.get('job_posted_at_timestamp')
                if ts:
                    posted_at = datetime.fromtimestamp(ts, tz=timezone.utc)

                raw = {
                    'title':       item.get('job_title', ''),
                    'company':     item.get('employer_name', ''),
                    'location':    f"{item.get('job_city', '')} {item.get('job_state', '')} India".strip(),
                    'skills':      item.get('job_required_skills') or [],
                    'salary':      f"{item.get('job_min_salary', '')} - {item.get('job_max_salary', '')}".strip(' -'),
                    'source':      item.get('job_publisher', 'jsearch').lower(),
                    'job_type':    'internship' if 'intern' in item.get('job_employment_type', '').lower() else 'fulltime',
                    'apply_url':   item.get('job_apply_link', ''),
                    'description': (item.get('job_description') or '')[:2000],
                    'posted_at':   posted_at,
                }
                if raw['title'] and raw['company']:
                    jobs.append(normalize_job(raw))

            logger.info(f"[JSearch] '{query}': {len(items)} jobs")
            time.sleep(0.5)

        except Exception as e:
            logger.error(f"[JSearch] Failed: {e}")

    logger.info(f"[JSearch] Total: {len(jobs)}")
    return jobs