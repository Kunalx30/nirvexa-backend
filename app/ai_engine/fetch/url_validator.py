"""
app/ai_engine/fetch/url_validator.py
Strict URL validation, URL normalization, deduplication, and multi-vector SSRF protection.
"""
import ipaddress
import re
import socket
import urllib.parse
from typing import List, Optional, Tuple, Set


BLOCKED_HOSTNAMES = {
    "localhost",
    "metadata.google.internal",
    "instance-data",
}

BLOCKED_DOMAIN_SUFFIXES = (
    ".local",
    ".internal",
    ".lan",
    ".localhost",
    ".corp",
    ".home",
)

TRACKING_QUERY_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
    "ref",
    "ref_src",
}


def normalize_url(url: str) -> Optional[str]:
    """
    Normalize URL:
    - Trim whitespace
    - Lowercase scheme and host
    - Remove URL fragment (#...)
    - Strip tracking query parameters
    - Remove trailing slashes for standard root paths
    """
    if not url or not isinstance(url, str):
        return None

    url = url.strip()
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return None

    if not parsed.scheme or not parsed.netloc:
        return None

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    # Clean query parameters
    query_params = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    filtered_params = [
        (k, v) for k, v in query_params if k.lower() not in TRACKING_QUERY_PARAMS
    ]
    new_query = urllib.parse.urlencode(filtered_params)

    # Normalize path (strip trailing slash if length > 1)
    path = parsed.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    return urllib.parse.urlunparse((scheme, netloc, path, parsed.params, new_query, ""))


def deduplicate_urls(urls: List[str]) -> List[str]:
    """Normalize and deduplicate URLs while preserving original order."""
    seen: Set[str] = set()
    deduped: List[str] = []

    for u in urls:
        norm = normalize_url(u)
        if norm and norm not in seen:
            seen.add(norm)
            deduped.append(norm)

    return deduped


def is_ip_blocked(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address) -> Tuple[bool, str]:
    """
    Check if an IP address falls within private, loopback, link-local,
    multicast, reserved, or cloud metadata ranges.
    """
    ip_str = str(ip_obj)

    # Cloud metadata (explicit 169.254.169.254)
    if ip_str == "169.254.169.254":
        return True, "Cloud metadata service IP blocked"

    # Loopback (127.0.0.0/8, ::1)
    if ip_obj.is_loopback:
        return True, "Loopback address blocked"

    # Private (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, fc00::/7)
    if ip_obj.is_private:
        return True, "Private subnet address blocked"

    # Link-local (169.254.0.0/16, fe80::/10)
    if ip_obj.is_link_local:
        return True, "Link-local address blocked"

    # Multicast (224.0.0.0/4, ff00::/8)
    if ip_obj.is_multicast:
        return True, "Multicast address blocked"

    # Reserved (0.0.0.0/8, 240.0.0.0/4, etc.)
    if ip_obj.is_reserved or ip_obj.is_unspecified:
        return True, "Reserved/unspecified address blocked"

    # Carrier-Grade NAT (100.64.0.0/10)
    if isinstance(ip_obj, ipaddress.IPv4Address):
        cgnat = ipaddress.IPv4Network("100.64.0.0/10")
        if ip_obj in cgnat:
            return True, "Carrier-Grade NAT address blocked"

    return False, ""


def validate_url(url: str, resolve_dns: bool = True) -> Tuple[bool, str, Optional[str]]:
    """
    Validates a URL against SSRF vulnerabilities.
    Returns: (is_valid: bool, reason: str, normalized_url: Optional[str])
    """
    if not url or not isinstance(url, str):
        return False, "Invalid or malformed URL", None

    url_str = url.strip()
    try:
        raw_parsed = urllib.parse.urlparse(url_str)
    except Exception:
        return False, "Invalid or malformed URL", None

    # 1. Check protocol before netloc
    if not raw_parsed.scheme:
        return False, "Missing URL scheme", None

    scheme_lower = raw_parsed.scheme.lower()
    if scheme_lower not in ("http", "https"):
        return False, f"Unsupported scheme: {raw_parsed.scheme}", None

    normalized = normalize_url(url_str)
    if not normalized:
        return False, "Invalid or malformed URL", None

    parsed = urllib.parse.urlparse(normalized)

    # 2. Extract host and port
    hostname = parsed.hostname
    if not hostname:
        return False, "Missing hostname in URL", None

    hostname_lower = hostname.lower()

    # 3. Check blocked hostnames
    if hostname_lower in BLOCKED_HOSTNAMES:
        return False, f"Blocked hostname: {hostname}", None

    # 4. Check blocked domain suffixes (.local, .internal, etc.)
    if any(hostname_lower.endswith(suffix) for suffix in BLOCKED_DOMAIN_SUFFIXES):
        return False, f"Internal domain suffix blocked: {hostname}", None

    # 5. Direct IP literal check
    try:
        ip_obj = ipaddress.ip_address(hostname_lower)
        is_blocked, reason = is_ip_blocked(ip_obj)
        if is_blocked:
            return False, f"SSRF violation: {reason} ({ip_obj})", None
    except ValueError:
        # Hostname is a domain name, not an IP literal
        pass

    # 6. DNS Resolution validation
    if resolve_dns:
        try:
            # Resolve all addresses (both IPv4 and IPv6)
            addr_info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
            if not addr_info:
                return False, f"DNS resolution failed for {hostname}", None

            for family, socktype, proto, canonname, sockaddr in addr_info:
                ip_str = sockaddr[0]
                ip_obj = ipaddress.ip_address(ip_str)
                is_blocked, reason = is_ip_blocked(ip_obj)
                if is_blocked:
                    return False, f"DNS resolved to blocked IP: {reason} ({ip_str})", None

        except (socket.gaierror, socket.herror) as e:
            return False, f"DNS resolution error for {hostname}: {e}", None
        except Exception as e:
            return False, f"Validation resolution error: {e}", None

    return True, "Valid", normalized
