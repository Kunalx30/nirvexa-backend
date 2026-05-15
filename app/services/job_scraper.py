import os
import requests
import time
import logging
import urllib3
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
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

    raw_job_type = (raw.get('job_type') or 'fulltime').strip().lower()
    if raw_job_type in {'full-time', 'full time', 'full_time'}:
        job_type = 'fulltime'
    elif 'contract' in raw_job_type:
        job_type = 'contract'
    elif 'part' in raw_job_type:
        job_type = 'parttime'
    elif 'intern' in raw_job_type:
        job_type = 'internship'
    elif 'remote' in raw_job_type:
        job_type = 'remote'
    elif 'freelance' in raw_job_type:
        job_type = 'freelance'
    else:
        job_type = raw_job_type

    title_lower = (raw.get('title') or '').lower()
    desc_lower  = (raw.get('description') or '').lower()
    source      = raw.get('source', '')
    
    FRESHER_KEYWORDS = {
        'fresher', 'entry level', 'entry-level', 'junior', '0-1 year',
        '0 to 1', 'no experience', 'graduate', 'trainee', 'campus',
        'associate', 'intern', 'apprentice', '0-2 year', 'recent grad',
    }
    
    is_fresher = raw.get('is_fresher', False) or (
    source in {'internshala', 'freshersworld', 'foundit'}
    or job_type in {'internship'}
    or any(kw in title_lower for kw in FRESHER_KEYWORDS)
    or any(kw in desc_lower  for kw in FRESHER_KEYWORDS)
)

    return {
        'title':       (raw.get('title') or '').strip(),
        'company':     (raw.get('company') or '').strip(),
        'location':    (raw.get('location') or 'Remote').strip(),
        'skills':      skills,
        'salary':      salary,
        'is_fresher': is_fresher,
        'source':      raw.get('source', 'unknown'),
        'job_type':    job_type,
        'apply_url':   raw.get('apply_url') or '',
        'description': raw.get('description') or '',
        'posted_at':   raw.get('posted_at') or datetime.now(timezone.utc),
        'expires_at':  datetime.now(timezone.utc) + timedelta(days=30),
    }


# ── STABLE WORKING SCRAPERS ───────────────────────────────────────────────────

def scrape_remotive() -> list[dict]:
    """Remotive.com — free REST API. Quality remote tech jobs, low competition."""
    url = "https://remotive.com/api/remote-jobs"
    try:
        resp = requests.get(url, timeout=15, headers={'User-Agent': 'NirVexa/1.0 Job Aggregator'})
        resp.raise_for_status()
        items = resp.json().get('jobs', [])
        jobs = []
        for item in items:
            posted_at = datetime.now(timezone.utc)
            try:
                posted_at = datetime.strptime(item.get('publication_date', ''), "%Y-%m-%dT%H:%M:%S")
            except Exception:
                pass
            jobs.append(normalize_job({
                'title':       item.get('title', ''),
                'company':     item.get('company_name', ''),
                'location':    item.get('candidate_required_location') or 'Remote',
                'skills':      item.get('tags') or [],
                'salary':      item.get('salary') or '',
                'source':      'remotive',
                'job_type':    'remote',
                'apply_url':   item.get('url', ''),
                'description': item.get('description', '')[:2000],
                'posted_at':   posted_at,
            }))
        logger.info(f"[Remotive] Fetched {len(jobs)} jobs")
        return jobs
    except Exception as e:
        logger.error(f"[Remotive] Failed: {e}")
        return []


def scrape_github_jobs() -> list[dict]:
    """Arbeitnow — free API. EU + remote tech jobs."""
    url = "https://www.arbeitnow.com/api/job-board-api"
    try:
        resp = requests.get(url, timeout=15, headers={'User-Agent': 'NirVexa/1.0 Job Aggregator'})
        resp.raise_for_status()
        items = resp.json().get('data', [])
        jobs = []
        for item in items:
            posted_at = datetime.now(timezone.utc)
            try:
                posted_at = datetime.fromtimestamp(item.get('created_at', 0), tz=timezone.utc)
            except Exception:
                pass
            jobs.append(normalize_job({
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
            }))
        logger.info(f"[Arbeitnow] Fetched {len(jobs)} jobs")
        return jobs
    except Exception as e:
        logger.error(f"[Arbeitnow] Failed: {e}")
        return []


def scrape_internshala() -> list[dict]:
    """Internshala — best Indian fresher + internship source."""
    base_url  = "https://internshala.com"
    headers   = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'}
    jobs      = []
    endpoints = [
        ("/internships",                     "internship"),
        ("/jobs/fresher-jobs",               "fulltime"),
        ("/internships/work-from-home-jobs", "remote"),
        ("/jobs/part-time-jobs",             "parttime"),
    ]

    for path, job_type in endpoints:
        try:
            resp = requests.get(base_url + path, headers=headers, timeout=15)
            resp.raise_for_status()
            soup  = BeautifulSoup(resp.text, 'lxml')
            cards = soup.select('.individual_internship')
            logger.info(f"[Internshala] {path} → found {len(cards)} cards")

            for card in cards:
                try:
                    title    = card.select_one('.job-internship-name, .profile')
                    company  = card.select_one('.company-name')
                    location = card.select_one('.location_link, .locations')
                    stipend  = card.select_one('.stipend, .salary')

                    # Broader link selector to catch both job and internship detail URLs
                    link = card.select_one(
                        'a.view_detail_button, '
                        'a[href*="/internship/detail"], '
                        'a[href*="/jobs/detail"]'
                    )

                    apply_url = ''
                    if link and link.get('href'):
                        href = link['href']
                        apply_url = href if href.startswith('http') else base_url + href

                    skill_tags = card.select('.round_tabs span, .skills span')
                    skills     = [s.get_text(strip=True) for s in skill_tags if s.get_text(strip=True)]

                    title_text    = title.get_text(strip=True)   if title    else ''
                    company_text  = company.get_text(strip=True)  if company  else ''
                    location_text = location.get_text(strip=True) if location else 'India'
                    stipend_text  = stipend.get_text(strip=True)  if stipend  else ''

                    if not title_text or not company_text:
                        continue

                    # Mark as fresher if it's an internship, part-time, WFH, or fresher job
                    is_fresher = job_type in {'internship', 'parttime', 'remote'} or 'fresher' in path

                    raw = {
                        'title':       title_text,
                        'company':     company_text,
                        'location':    location_text,
                        'skills':      skills,
                        'salary':      stipend_text,
                        'source':      'internshala',
                        'job_type':    job_type,
                        'apply_url':   apply_url,
                        'description': '',
                        'posted_at':   datetime.now(timezone.utc),
                        'is_fresher':  is_fresher,
                    }
                    jobs.append(normalize_job(raw))

                except Exception as e:
                    logger.warning(f"[Internshala] Card parse error: {e}")
                    continue

            time.sleep(1)

        except Exception as e:
            logger.error(f"[Internshala] Failed on {path}: {e}")
            continue

    logger.info(f"[Internshala] Total fetched: {len(jobs)}")
    return jobs

def scrape_indian_startups_greenhouse() -> list[dict]:
    """
    Greenhouse ATS — verified slugs only.
    Quality startups, direct applications, low applicant flood.
    """
    companies = [
        ('postman',     'Postman'),
        ('druva',       'Druva'),
        ('figma',       'Figma'),
        ('vercel',      'Vercel'),
        ('groww',       'Groww'),
    ]
    jobs = []
    for slug, company_name in companies:
        try:
            url  = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true"
            resp = requests.get(url, timeout=10, headers={'User-Agent': 'NirVexa/1.0 Job Aggregator'})
            if resp.status_code != 200:
                logger.warning(f"[GH-India] {company_name}: HTTP {resp.status_code} — skipping")
                time.sleep(0.3)
                continue
            items = resp.json().get('jobs', [])
            for item in items:
                offices  = item.get('offices') or []
                location = ', '.join([o.get('name', '') for o in offices]) or 'India'
                depts    = item.get('departments') or []
                skills   = [d.get('name', '') for d in depts if d.get('name')]
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
                    'posted_at':   datetime.now(timezone.utc),
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


def scrape_lever_companies() -> list[dict]:
    """Lever ATS — quality companies, direct applications."""
    companies = [('cred', 'CRED')]
    jobs = []
    for slug, company_name in companies:
        try:
            url  = f"https://api.lever.co/v0/postings/{slug}?mode=json"
            resp = requests.get(url, timeout=15, headers={'User-Agent': 'NirVexa/1.0 Job Aggregator'})
            resp.raise_for_status()
            items = resp.json()
            if not isinstance(items, list):
                continue
            for item in items:
                categories = item.get('categories', {})
                location   = categories.get('location') or 'India'
                if isinstance(location, list):
                    location = ', '.join(location)
                team       = categories.get('team', '')
                commitment = categories.get('commitment', 'fulltime').lower()
                if 'intern' in commitment:
                    job_type = 'internship'
                elif 'contract' in commitment:
                    job_type = 'contract'
                elif 'part' in commitment:
                    job_type = 'parttime'
                else:
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
                    'posted_at': datetime.fromtimestamp(item.get('createdAt', 0) / 1000, tz=timezone.utc) if item.get('createdAt') else datetime.now(timezone.utc),
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


def scrape_indian_startups_lever() -> list[dict]:
    """
    Lever — verified Indian startup slugs.
    Direct apply on company site = less competition vs Naukri.
    """
    companies = [
        ('meesho',       'Meesho'),
        ('paytm',        'Paytm'),
        ('freshworks',   'Freshworks'),
    ]
    jobs = []
    for slug, company_name in companies:
        try:
            url  = f"https://api.lever.co/v0/postings/{slug}?mode=json"
            resp = requests.get(url, timeout=10, headers={'User-Agent': 'NirVexa/1.0 Job Aggregator'})
            if resp.status_code != 200:
                logger.warning(f"[Lever-India] {company_name}: {resp.status_code} — skip")
                time.sleep(0.3)
                continue
            items = resp.json()
            if not isinstance(items, list):
                continue
            count = 0
            for item in items:
                categories = item.get('categories', {})
                location   = categories.get('location') or 'India'
                if isinstance(location, list):
                    location = ', '.join(location)
                team       = categories.get('team', '')
                commitment = (categories.get('commitment') or 'fulltime').lower()
                if 'intern' in commitment:
                    job_type = 'internship'
                elif 'contract' in commitment:
                    job_type = 'contract'
                elif 'part' in commitment:
                    job_type = 'parttime'
                else:
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
                    'posted_at': datetime.fromtimestamp(item.get('createdAt', 0) / 1000, tz=timezone.utc) if item.get('createdAt') else datetime.now(timezone.utc),
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


def scrape_jsearch_india() -> list[dict]:
    """JSearch — LinkedIn/Indeed/Naukri aggregator. Daily guard protects 500/month limit."""
    import json
    api_key = os.getenv('RAPIDAPI_KEY')
    if not api_key:
        logger.error("[JSearch] No RAPIDAPI_KEY in .env")
        return []

    guard_file = "jsearch_guard.json"
    today      = str(datetime.now(timezone.utc).date())
    try:
        if os.path.exists(guard_file):
            with open(guard_file) as f:
                guard = json.load(f)
            if guard.get("date") == today:
                logger.info("[JSearch] Already ran today — skipping.")
                return []
    except Exception:
        pass

    queries = [
        "fresher software engineer jobs India 0-1 years",
        "entry level python developer jobs India",
        "junior data analyst jobs India no experience",
        "campus hire software developer India",
        "graduate trainee engineer jobs India",
    ]

    jobs    = []
    headers = {"X-RapidAPI-Key": api_key, "X-RapidAPI-Host": "jsearch.p.rapidapi.com"}

    for query in queries:
        try:
            resp = requests.get(
                "https://jsearch.p.rapidapi.com/search",
                headers=headers,
                params={"query": query, "page": "1", "num_pages": "1", "date_posted": "today", "country": "in"},
                timeout=15
            )
            if resp.status_code == 429:
                logger.warning("[JSearch] Rate limited — saving guard and stopping")
                break
            if resp.status_code != 200:
                logger.warning(f"[JSearch] {query}: HTTP {resp.status_code} — {resp.text[:200]}")
                continue
            items = resp.json().get('data', [])
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

    if jobs:  # only save guard if we actually got results
        try:
            with open(guard_file, "w") as f:
                json.dump({"date": today}, f)
        except Exception:
            pass

    logger.info(f"[JSearch] Total: {len(jobs)}")
    return jobs


# ── NEW QUALITY SCRAPERS ──────────────────────────────────────────────────────

def scrape_remoteok() -> list[dict]:
    """RemoteOK — free JSON API. Quality remote jobs, direct company posts."""
    url = "https://remoteok.com/api"
    try:
        resp  = requests.get(url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'
        })
        resp.raise_for_status()
        items = [i for i in resp.json() if isinstance(i, dict) and i.get('position')]
        jobs  = []
        for item in items:
            posted_at = datetime.now(timezone.utc)
            try:
                posted_at = datetime.fromtimestamp(item.get('epoch', 0), tz=timezone.utc)
            except Exception:
                pass
            jobs.append(normalize_job({
                'title':       item.get('position', ''),
                'company':     item.get('company', ''),
                'location':    item.get('location') or 'Remote',
                'skills':      item.get('tags') or [],
                'salary':      item.get('salary') or '',
                'source':      'remoteok',
                'job_type':    'remote',
                'apply_url':   item.get('url') or f"https://remoteok.com/remote-jobs/{item.get('id', '')}",
                'description': (item.get('description') or '')[:2000],
                'posted_at':   posted_at,
            }))
        logger.info(f"[RemoteOK] Fetched {len(jobs)} jobs")
        return jobs
    except Exception as e:
        logger.error(f"[RemoteOK] Failed: {e}")
        return []


def scrape_weworkremotely() -> list[dict]:
    """
    We Work Remotely — RSS feeds.
    Curated remote jobs, direct company posts, quality over quantity.
    Fixed: namespace stripping for clean XML parse.
    """
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime

    feeds = [
        ("https://weworkremotely.com/categories/remote-programming-jobs.rss",    "Programming"),
        ("https://weworkremotely.com/categories/remote-devops-sysadmin-jobs.rss", "DevOps"),
        ("https://weworkremotely.com/categories/remote-design-jobs.rss",          "Design"),
        ("https://weworkremotely.com/categories/remote-product-jobs.rss",         "Product"),
    ]

    jobs    = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'}

    for feed_url, category in feeds:
        try:
            resp = requests.get(feed_url, timeout=10, headers=headers)
            if resp.status_code != 200:
                logger.warning(f"[WWR] {category}: HTTP {resp.status_code}")
                continue

            # Strip XML namespaces to simplify parsing
            content = resp.content.decode('utf-8', errors='replace')
            content = content.replace(' xmlns=', ' xmlnsx=')
            root    = ET.fromstring(content.encode('utf-8'))
            channel = root.find('channel')
            if not channel:
                continue

            count = 0
            for item in channel.findall('item'):
                try:
                    title_raw = (item.findtext('title') or '').strip()
                    link      = (item.findtext('link')  or '').strip()
                    pubdate   = item.findtext('pubDate') or ''

                    # WWR format: "Company Name: Job Title"
                    company, title = '', title_raw
                    if ': ' in title_raw:
                        parts   = title_raw.split(': ', 1)
                        company = parts[0].strip()
                        title   = parts[1].strip()

                    # Remove trailing " at Anywhere"
                    if ' at ' in title:
                        title = title.rsplit(' at ', 1)[0].strip()

                    posted_at = datetime.now(timezone.utc)
                    try:
                        if pubdate:
                            posted_at = parsedate_to_datetime(pubdate)
                    except Exception:
                        pass

                    if title and company:
                        jobs.append(normalize_job({
                            'title':       title,
                            'company':     company,
                            'location':    'Remote',
                            'skills':      [category],
                            'salary':      '',
                            'source':      'weworkremotely',
                            'job_type':    'remote',
                            'apply_url':   link,
                            'description': '',
                            'posted_at':   posted_at,
                        }))
                        count += 1
                except Exception:
                    continue

            logger.info(f"[WWR] {category}: {count} jobs")
            time.sleep(0.5)

        except Exception as e:
            logger.error(f"[WWR] {category} failed: {e}")

    logger.info(f"[WeWorkRemotely] Total: {len(jobs)}")
    return jobs


def scrape_workingnomads() -> list[dict]:
    """Working Nomads — free JSON API. Curated remote jobs, quality hiring."""
    categories = ['developer', 'data', 'devops', 'design', 'product']
    jobs       = []

    for cat in categories:
        try:
            url  = f"https://www.workingnomads.com/api/exposed_jobs/?category={cat}"
            resp = requests.get(url, timeout=10, headers={'User-Agent': 'NirVexa/1.0 Job Aggregator'})
            if resp.status_code != 200:
                continue
            items = resp.json()
            if not isinstance(items, list):
                continue
            for item in items:
                posted_at = datetime.now(timezone.utc)
                try:
                    posted_at = datetime.strptime(
                        item.get('pub_date', '')[:19], "%Y-%m-%dT%H:%M:%S"
                    ).replace(tzinfo=timezone.utc)
                except Exception:
                    pass
                jobs.append(normalize_job({
                    'title':       item.get('title', ''),
                    'company':     item.get('company', ''),
                    'location':    item.get('location') or 'Remote',
                    'skills':      [cat],
                    'salary':      '',
                    'source':      'workingnomads',
                    'job_type':    'remote',
                    'apply_url':   item.get('url', ''),
                    'description': (item.get('description') or '')[:2000],
                    'posted_at':   posted_at,
                }))
            logger.info(f"[WorkingNomads] {cat}: {len(items)} jobs")
            time.sleep(0.3)
        except Exception as e:
            logger.error(f"[WorkingNomads] {cat} failed: {e}")

    logger.info(f"[WorkingNomads] Total: {len(jobs)}")
    return jobs


def scrape_himalayas() -> list[dict]:
    """Himalayas — free JSON API. Quality remote tech jobs, no auth."""
    url = "https://himalayas.app/jobs/api?limit=50"
    try:
        resp = requests.get(url, timeout=15, headers={'User-Agent': 'NirVexa/1.0 Job Aggregator'})
        resp.raise_for_status()
        items = resp.json().get('jobs', [])
        jobs = []
        for item in items:
            posted_at = datetime.now(timezone.utc)
            try:
                posted_at = datetime.strptime(
                    item.get('createdAt', '')[:19], "%Y-%m-%dT%H:%M:%S"
                ).replace(tzinfo=timezone.utc)
            except Exception:
                pass
            skills = item.get('skills') or []
            if isinstance(skills, list):
                skills = [s.get('title', '') if isinstance(s, dict) else s for s in skills]

            location = item.get('locationRestrictions') or 'Remote'
            if isinstance(location, list):
                parts = []
                for loc in location:
                    if isinstance(loc, dict):
                        parts.append(loc.get('name') or loc.get('title') or loc.get('country') or '')
                    else:
                        parts.append(str(loc))
                location = ', '.join(part for part in parts if part) or 'Remote'
            elif isinstance(location, dict):
                location = location.get('name') or location.get('title') or location.get('country') or 'Remote'

            jobs.append(normalize_job({
                'title':       item.get('title', ''),
                'company':     item.get('company', {}).get('name', '') if isinstance(item.get('company'), dict) else '',
                'location':    location,
                'skills':      skills,
                'salary':      '',
                'source':      'himalayas',
                'job_type':    'remote',
                'apply_url':   item.get('applicationLink', ''),
                'description': (item.get('description') or '')[:2000],
                'posted_at':   posted_at,
            }))
        logger.info(f"[Himalayas] Fetched {len(jobs)} jobs")
        return jobs
    except Exception as e:
        logger.error(f"[Himalayas] Failed: {e}")
        return []


def scrape_aijobs() -> list[dict]:
    """
    AI Jobs Net — AI/ML/Data Science niche board.
    Less competition, better fit for NirVexa data science users.
    Fixed: tries multiple endpoints.
    """
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime

    urls = [
        "https://aijobs.net/feed/",
        "https://aijobs.net/rss/",
    ]

    for url in urls:
        try:
            resp = requests.get(url, timeout=10, headers={'User-Agent': 'NirVexa/1.0 Job Aggregator'})
            if resp.status_code != 200:
                continue

            root    = ET.fromstring(resp.content)
            channel = root.find('channel')
            if not channel:
                continue

            jobs = []
            for item in channel.findall('item'):
                title_el   = item.find('title')
                link_el    = item.find('link')
                desc_el    = item.find('description')
                pubdate_el = item.find('pubDate')

                title   = (title_el.text   or '').strip() if title_el   else ''
                link    = (link_el.text    or '').strip() if link_el    else ''
                desc    = (desc_el.text    or '').strip() if desc_el    else ''

                company = ''
                if ' at ' in title:
                    parts   = title.rsplit(' at ', 1)
                    title   = parts[0].strip()
                    company = parts[1].strip()

                posted_at = datetime.now(timezone.utc)
                try:
                    if pubdate_el and pubdate_el.text:
                        posted_at = parsedate_to_datetime(pubdate_el.text)
                except Exception:
                    pass

                if title:
                    jobs.append(normalize_job({
                        'title':       title,
                        'company':     company or 'Unknown',
                        'location':    'Remote',
                        'skills':      ['AI', 'ML', 'Data Science'],
                        'salary':      '',
                        'source':      'aijobs',
                        'job_type':    'fulltime',
                        'apply_url':   link,
                        'description': desc[:2000],
                        'posted_at':   posted_at,
                    }))

            logger.info(f"[AIJobs] Fetched {len(jobs)} jobs")
            return jobs

        except Exception as e:
            logger.error(f"[AIJobs] {url} failed: {e}")
            continue

    logger.warning("[AIJobs] All endpoints failed")
    return []


def scrape_instahyre() -> list[dict]:
    """
    Instahyre — Indian startup jobs, JSON embedded in page.
    Better than Naukri for fresher/startup roles.
    """
    roles = [
        'software-engineer', 'data-scientist', 'python-developer',
        'frontend-developer', 'backend-developer', 'android-developer',
    ]
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json',
    }
    jobs = []
    for role in roles:
        try:
            url  = f"https://www.instahyre.com/api/v1/opportunity/?format=json&search={role}&location=India&page=1"
            resp = requests.get(url, timeout=15, headers=headers)
            if resp.status_code != 200:
                logger.warning(f"[Instahyre] {role}: HTTP {resp.status_code}")
                time.sleep(1)
                continue
            data  = resp.json()
            items = data.get('results') or data.get('opportunities') or []
            if not isinstance(items, list):
                continue
            count = 0
            for item in items:
                company_info = item.get('employer') or item.get('company') or {}
                company_name = company_info.get('name', '') if isinstance(company_info, dict) else ''
                skills_raw   = item.get('skills') or []
                skills       = [s.get('name', '') if isinstance(s, dict) else s for s in skills_raw]
                location     = item.get('location') or item.get('city') or 'India'
                if isinstance(location, list):
                    location = ', '.join(location)
                posted_at = datetime.now(timezone.utc)
                try:
                    posted_at = datetime.strptime(
                        item.get('created_at', '')[:19], "%Y-%m-%dT%H:%M:%S"
                    ).replace(tzinfo=timezone.utc)
                except Exception:
                    pass
                title = item.get('designation') or item.get('title') or ''
                if title and company_name:
                    jobs.append(normalize_job({
                        'title':       title,
                        'company':     company_name,
                        'location':    location,
                        'skills':      skills,
                        'salary':      str(item.get('salary_range', '')),
                        'source':      'instahyre',
                        'job_type':    'fulltime',
                        'apply_url':   f"https://www.instahyre.com/candidate/opportunities/{item.get('id', '')}",
                        'description': (item.get('description') or '')[:2000],
                        'posted_at':   posted_at,
                    }))
                    count += 1
            logger.info(f"[Instahyre] {role}: {count} jobs")
            time.sleep(1)
        except Exception as e:
            logger.error(f"[Instahyre] {role} failed: {e}")
    logger.info(f"[Instahyre] Total: {len(jobs)}")
    return jobs

def scrape_adzuna_by_city() -> list[dict]:
    """
    Adzuna India — city-based IT job scraping.
    Fixed: using category=it-jobs filter, no role restriction.
    10 cities × 50 jobs = 500 quality Indian jobs/day.
    """
    app_id  = os.getenv('ADZUNA_APP_ID')
    app_key = os.getenv('ADZUNA_APP_KEY')

    if not app_id or not app_key:
        logger.error("[Adzuna-City] Missing ADZUNA_APP_ID or ADZUNA_APP_KEY in .env")
        return []

    cities = [
        'Bangalore', 'Hyderabad', 'Pune',    'Delhi',
        'Mumbai',    'Chennai',   'Noida',   'Gurgaon',
        'Kolkata',   'Ahmedabad'
    ]

    jobs = []
    for city in cities:
        try:
            url  = (
                f"https://api.adzuna.com/v1/api/jobs/in/search/1"
                f"?app_id={app_id}&app_key={app_key}"
                f"&results_per_page=50"
                f"&where={city}"
                f"&category=it-jobs"
                f"&content-type=application/json"
            )
            resp = requests.get(url, timeout=15, headers={'User-Agent': 'NirVexa/1.0 Job Aggregator'})
            if resp.status_code != 200:
                logger.warning(f"[Adzuna-City] {city}: HTTP {resp.status_code}")
                continue

            items = resp.json().get('results', [])
            for item in items:
                posted_at = datetime.now(timezone.utc)
                try:
                    posted_at = datetime.strptime(
                        item.get('created', '')[:19], "%Y-%m-%dT%H:%M:%S"
                    ).replace(tzinfo=timezone.utc)
                except Exception:
                    pass

                jobs.append(normalize_job({
                    'title':       item.get('title', ''),
                    'company':     item.get('company', {}).get('display_name', ''),
                    'location':    city,
                    'skills':      [item.get('category', {}).get('label', '')],
                    'salary':      f"{item.get('salary_min', '')} - {item.get('salary_max', '')}".strip(' -'),
                    'source':      'adzuna',
                    'job_type':    'fulltime',
                    'apply_url':   item.get('redirect_url', ''),
                    'description': (item.get('description') or '')[:2000],
                    'posted_at':   posted_at,
                }))

            logger.info(f"[Adzuna-City] {city}: {len(items)} jobs")
            time.sleep(0.5)

        except Exception as e:
            logger.error(f"[Adzuna-City] {city} failed: {e}")

    logger.info(f"[Adzuna-City] Total: {len(jobs)}")
    return jobs


def scrape_adzuna_by_role() -> list[dict]:
    """
    Adzuna India — role-based scraping.
    10 high-demand IT roles × 20 jobs = 200 quality jobs/day.
    """
    app_id  = os.getenv('ADZUNA_APP_ID')
    app_key = os.getenv('ADZUNA_APP_KEY')

    if not app_id or not app_key:
        logger.error("[Adzuna-Role] Missing ADZUNA_APP_ID or ADZUNA_APP_KEY in .env")
        return []

    roles = [
        'software engineer',
        'data scientist',
        'devops engineer',
        'frontend developer',
        'backend developer',
        'android developer',
        'cybersecurity analyst',
        'full stack developer',
        'business analyst',
        'qa engineer',
    ]

    jobs = []
    for role in roles:
        try:
            url  = (
                f"https://api.adzuna.com/v1/api/jobs/in/search/1"
                f"?app_id={app_id}&app_key={app_key}"
                f"&results_per_page=20"
                f"&what={role.replace(' ', '+')}"
                f"&content-type=application/json"
            )
            resp = requests.get(url, timeout=15, headers={'User-Agent': 'NirVexa/1.0 Job Aggregator'})
            if resp.status_code != 200:
                logger.warning(f"[Adzuna-Role] {role}: HTTP {resp.status_code}")
                continue

            items = resp.json().get('results', [])
            for item in items:
                posted_at = datetime.now(timezone.utc)
                try:
                    posted_at = datetime.strptime(
                        item.get('created', '')[:19], "%Y-%m-%dT%H:%M:%S"
                    ).replace(tzinfo=timezone.utc)
                except Exception:
                    pass

                location = item.get('location', {}).get('display_name', 'India')

                jobs.append(normalize_job({
                    'title':       item.get('title', ''),
                    'company':     item.get('company', {}).get('display_name', ''),
                    'location':    location,
                    'skills':      [role],
                    'salary':      f"{item.get('salary_min', '')} - {item.get('salary_max', '')}".strip(' -'),
                    'source':      'adzuna',
                    'job_type':    'fulltime',
                    'apply_url':   item.get('redirect_url', ''),
                    'description': (item.get('description') or '')[:2000],
                    'posted_at':   posted_at,
                }))

            logger.info(f"[Adzuna-Role] '{role}': {len(items)} jobs")
            time.sleep(0.5)

        except Exception as e:
            logger.error(f"[Adzuna-Role] '{role}' failed: {e}")

    logger.info(f"[Adzuna-Role] Total: {len(jobs)}")
    return jobs


def scrape_naukri_india() -> list[dict]:
    """
    Naukri.com RSS — replaces broken SimplyHired India.
    Stable XML feed, top Indian job board, high volume.
    """
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime


    # Naukri's API needs auth now, so use their public RSS instead
    rss_searches = [
        ("software-engineer",  "bangalore"),
        ("python-developer",   "hyderabad"),
        ("data-scientist",     "pune"),
        ("java-developer",     "delhi"),
        ("frontend-developer", "mumbai"),
        ("devops-engineer",    "chennai"),
        ("android-developer",  "noida"),
        ("full-stack-developer", "gurgaon"),
    ]

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'
    }

    jobs = []
    for role, city in rss_searches:
        try:
            # Naukri search page (scraping public listing)
            url  = f"https://www.naukri.com/{role}-jobs-in-{city}"
            resp = requests.get(url, headers=headers, timeout=15, verify=False)
            if resp.status_code != 200:
                logger.warning(f"[Naukri] {role}/{city}: HTTP {resp.status_code}")
                time.sleep(1)
                continue

            soup  = BeautifulSoup(resp.text, 'lxml')

            # Naukri embeds job data in script tags as JSON
            import json, re
            script_tags = soup.find_all('script', type='application/ld+json')
            found = 0
            for tag in script_tags:
                try:
                    raw_text = tag.string or ''
                    data = json.loads(raw_text)
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        if item.get('@type') == 'JobPosting':
                            org     = item.get('hiringOrganization', {})
                            company = org.get('name', '') if isinstance(org, dict) else ''
                            loc_obj = item.get('jobLocation', {})
                            if isinstance(loc_obj, list):
                                loc_obj = loc_obj[0] if loc_obj else {}
                            address = loc_obj.get('address', {}) if isinstance(loc_obj, dict) else {}
                            location = (
                                address.get('addressLocality') or
                                address.get('addressRegion') or
                                city.title()
                            )
                            posted_raw = item.get('datePosted', '')
                            posted_at  = datetime.now(timezone.utc)
                            try:
                                posted_at = datetime.fromisoformat(posted_raw.replace('Z', '+00:00'))
                            except Exception:
                                pass

                            if item.get('title') and company:
                                jobs.append(normalize_job({
                                    'title':       item['title'],
                                    'company':     company,
                                    'location':    location,
                                    'skills':      item.get('skills', []) if isinstance(item.get('skills'), list) else [],
                                    'salary':      '',
                                    'source':      'naukri',
                                    'job_type':    'fulltime',
                                    'apply_url':   item.get('url', ''),
                                    'description': (item.get('description') or '')[:2000],
                                    'posted_at':   posted_at,
                                }))
                                found += 1
                except Exception:
                    continue

            logger.info(f"[Naukri] {role} in {city}: {found} jobs")
            time.sleep(2)

        except Exception as e:
            logger.error(f"[Naukri] {role}/{city} failed: {e}")

    logger.info(f"[Naukri] Total: {len(jobs)}")
    return jobs



#freshr targetting jobs 
def scrape_freshersworld() -> list[dict]:
    """Freshersworld — India's largest fresher-only job board."""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'}
    jobs = []
    
    urls = [
        ("https://www.freshersworld.com/jobs/freshers", "fulltime"),
        ("https://www.freshersworld.com/jobs/internship", "internship"),
    ]
    
    for url, job_type in urls:
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'lxml')
            cards = soup.select('.job-container, .job-box, article.job')
            
            for card in cards:
                title    = card.select_one('.job-title, h3, h2')
                company  = card.select_one('.company-name, .employer')
                location = card.select_one('.location, .job-location')
                link     = card.select_one('a[href*="/jobs/"]')
                
                apply_url = ''
                if link and link.get('href'):
                    href = link['href']
                    apply_url = href if href.startswith('http') else 'https://www.freshersworld.com' + href
                
                raw = {
                    'title':    title.get_text(strip=True)    if title    else '',
                    'company':  company.get_text(strip=True)  if company  else '',
                    'location': location.get_text(strip=True) if location else 'India',
                    'skills':   [],
                    'salary':   '',
                    'source':   'freshersworld',
                    'job_type': job_type,
                    'apply_url': apply_url,
                    'description': '',
                    'posted_at': datetime.now(timezone.utc),
                }
                if raw['title'] and raw['company']:
                    jobs.append(normalize_job(raw))
        except Exception as e:
            logger.error(f"[Freshersworld] {url} failed: {e}")
    
    logger.info(f"[Freshersworld] Total: {len(jobs)}")
    return jobs


def scrape_foundit_freshers() -> list[dict]:
    """
    Foundit (formerly Monster India) — public JSON API.
    Supports fresher experience filter directly in URL.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Referer': 'https://www.foundit.in/',
    }

    searches = [
        ('software engineer',  'Bangalore'),
        ('python developer',   'Hyderabad'),
        ('data analyst',       'Pune'),
        ('frontend developer', 'Delhi'),
        ('java developer',     'Mumbai'),
    ]

    jobs = []
    for role, city in searches:
        try:
            resp = requests.get(
                "https://www.foundit.in/middleware/jobsearch/v1/search",
                headers=headers,
                params={
                    "query":              role,
                    "locationPreferences": city,
                    "experienceRanges":   "0-1",   # 0-1 years = fresher
                    "pageNo":             1,
                    "limit":              20,
                },
                timeout=15,
            )
            if resp.status_code != 200:
                logger.warning(f"[Foundit] {role}/{city}: HTTP {resp.status_code}")
                time.sleep(1)
                continue

            items = resp.json().get('jobSearchResponse', {}).get('data', [])
            for item in items:
                posted_at = datetime.now(timezone.utc)
                try:
                    posted_at = datetime.fromtimestamp(
                        item.get('postedDate', 0) / 1000, tz=timezone.utc
                    )
                except Exception:
                    pass

                jobs.append(normalize_job({
                    'title':       item.get('designation', ''),
                    'company':     item.get('companyName', ''),
                    'location':    city,
                    'skills':      item.get('keySkills', []) or [],
                    'salary':      item.get('salary', ''),
                    'source':      'foundit',
                    'job_type':    'fulltime',
                    'apply_url':   f"https://www.foundit.in/job/{item.get('jobId', '')}",
                    'description': (item.get('jobDescription') or '')[:2000],
                    'posted_at':   posted_at,
                }))

            logger.info(f"[Foundit] {role}/{city}: {len(items)} jobs")
            time.sleep(0.8)

        except Exception as e:
            logger.error(f"[Foundit] {role}/{city} failed: {e}")

    logger.info(f"[Foundit] Total: {len(jobs)}")
    return jobs


def scrape_greenhouse_fresher_companies() -> list[dict]:
    """
    Greenhouse — companies known for strong fresher/graduate hiring.
    Different from your existing Indian startup list.
    """
    companies = [
        ('thoughtworks',   'ThoughtWorks'),
        ('razorpay',       'Razorpay'),
        ('browserstack',   'BrowserStack'),
        ('hasura',         'Hasura'),
        ('setu',           'Setu'),
        ('clarisights',    'Clarisights'),
        ('nilenso',        'Nilenso'),
    ]
    jobs = []
    for slug, company_name in companies:
        try:
            url  = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true"
            resp = requests.get(url, timeout=10, headers={'User-Agent': 'NirVexa/1.0'})
            if resp.status_code != 200:
                time.sleep(0.3)
                continue
            items = resp.json().get('jobs', [])
            for item in items:
                # Only include junior/fresher-looking titles
                title_lower = item.get('title', '').lower()
                if any(kw in title_lower for kw in [
                    'junior', 'fresher', 'entry', 'graduate', 'intern',
                    'trainee', 'associate', '0-1', '0 - 1'
                ]):
                    offices  = item.get('offices') or []
                    location = ', '.join([o.get('name', '') for o in offices]) or 'India'
                    jobs.append(normalize_job({
                        'title':       item.get('title', ''),
                        'company':     company_name,
                        'location':    location,
                        'skills':      [],
                        'salary':      '',
                        'source':      'greenhouse',
                        'job_type':    'fulltime',
                        'apply_url':   item.get('absolute_url', ''),
                        'description': (item.get('content') or '')[:2000],
                        'posted_at':   datetime.now(timezone.utc),
                    }))
            logger.info(f"[GH-Fresher] {company_name}: added fresher-tagged jobs")
            time.sleep(0.4)
        except Exception as e:
            logger.error(f"[GH-Fresher] {company_name}: {e}")
    
    logger.info(f"[GH-Fresher] Total: {len(jobs)}")
    return jobs


def scrape_wellfound_freshers() -> list[dict]:
    """
    Wellfound (AngelList) — startup jobs, many entry-level.
    Uses public GraphQL — no auth needed for basic listings.
    """
    url = "https://wellfound.com/role/l/software-engineer/india"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml',
    }
    jobs = []
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            logger.warning(f"[Wellfound] HTTP {resp.status_code}")
            return []
        
        soup = BeautifulSoup(resp.text, 'lxml')
        import json, re
        
        # Wellfound embeds data in __NEXT_DATA__
        script = soup.find('script', id='__NEXT_DATA__')
        if script and script.string:
            data = json.loads(script.string)
            # Drill into the nested structure
            try:
                listings = (
                    data['props']['pageProps']
                       .get('searchResults', {})
                       .get('startupRoles', {})
                       .get('edges', [])
                )
                for edge in listings:
                    node    = edge.get('node', {})
                    startup = node.get('startup', {})
                    
                    title   = node.get('title', '')
                    company = startup.get('name', '')
                    
                    jobs.append(normalize_job({
                        'title':       title,
                        'company':     company,
                        'location':    node.get('locationNames', ['India'])[0] if node.get('locationNames') else 'India',
                        'skills':      node.get('skills', []) or [],
                        'salary':      '',
                        'source':      'wellfound',
                        'job_type':    'fulltime',
                        'apply_url':   f"https://wellfound.com/jobs/{node.get('id', '')}",
                        'description': (node.get('description') or '')[:2000],
                        'posted_at':   datetime.now(timezone.utc),
                    }))
            except (KeyError, TypeError) as e:
                logger.warning(f"[Wellfound] Data parse failed: {e}")
    
    except Exception as e:
        logger.error(f"[Wellfound] Failed: {e}")
    
    logger.info(f"[Wellfound] Total: {len(jobs)}")
    return jobs