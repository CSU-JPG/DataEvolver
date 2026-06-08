"""High-speed Bing image scraping & downloading."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import ssl
import time
from collections import Counter, defaultdict
from typing import (
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)
from urllib.parse import quote, urlparse

import aiohttp
import requests
from aiohttp import ClientTimeout
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

__all__ = [
    "async_fetch_image_urls",
    "async_download_urls",
    "async_fetch_and_download",
    "fetch_image_urls",
    "fetch_and_download",
    "make_session",
]

_DEFAULT_STEP = 57
_DEFAULT_CHUNK = 65536

MIN_VALID_HTML_LEN = 8_000
MAX_SEEN_CAP = 3_000
_PAGE_DELAY_MIN = 0.8
_PAGE_DELAY_MAX = 2.5
_CHALLENGE_BACKOFF_MIN = 4.0
_CHALLENGE_BACKOFF_MAX = 10.0
_HOST_COOLDOWN_MAX = 60.0
_HOST_BAN_THRESHOLD = 15      
_HOST_BAN_DURATION = 90.0  

_HOST_RAPID_FAIL_THRESHOLD = 3
_HOST_RAPID_FAIL_DURATION = 45.0

_DOWNLOAD_TOTAL_TIMEOUT = 12.0  
_DOWNLOAD_CONNECT_TIMEOUT = 4.0 
_DOWNLOAD_READ_TIMEOUT = 8.0 

_MIN_IMAGE_BYTES = 1_024

_BING_STATIC_COOKIES = {
    "SRCHLANG": "en",
    "SRCHD": "AF=NOFORM",
    "_EDGE_V": "1",
    "SRCHUSR": f"DOB={time.strftime('%Y%m%d')}",
}

_UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 "
    "Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) "
    "Gecko/20100101 Firefox/123.0",
]

_DOWNLOAD_HEADERS_BASE = {
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Cache-Control": "no-cache",
}

_FETCH_HEADERS_BASE = {
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

_KNOWN_BLOCKER_PATTERNS: Set[str] = {
    "shutterstock.com",
    "gettyimages.com",
    "istockphoto.com",
    "alamy.com",
    "dreamstime.com",
    "123rf.com",
    "depositphotos.com",
    "vectorstock.com",
    "bigstockphoto.com",
    "canstockphoto.com",
    "adobestock.com",
    "stock.adobe.com",
}

_CDN_HOSTS: Set[str] = {
    "wp.com",
    "wordpress.com",
    "cloudinary.com",
    "imgix.net",
    "fastly.net",
    "cloudfront.net",
    "akamaized.net",
    "googleusercontent.com",
    "ytimg.com",
    "twimg.com",
    "fbcdn.net",
    "cdninstagram.com",
    "pinimg.com",
    "media.tumblr.com",
    "staticflickr.com",
}

_IRRELEVANT_DOMAINS: Set[str] = {
    "yelp.com",
    "tripadvisor.com",
    "bakingo.com",
    "ecayonline.com",
    "zomato.com",
    "opentable.com",
    "grubhub.com",
    "doordash.com",
    "ubereats.com",
    "foodpanda.com",
    "swiggy.com",
}


def _pick_ua() -> str:
    """Return a random User-Agent from the pool."""
    return random.choice(_UA_POOL)


def _is_blocked_host(host: str) -> bool:
    return any(p in host for p in _KNOWN_BLOCKER_PATTERNS)


def _is_cdn_host(host: str) -> bool:
    return any(p in host for p in _CDN_HOSTS)


def _per_host_limit(host: str, default: int = 4) -> int:
    """CDN hosts get 3× the default concurrency limit."""
    return default * 3 if _is_cdn_host(host) else default


def _is_irrelevant_url(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return any(d in host for d in _IRRELEVANT_DOMAINS)


def _clean_query(raw: str) -> str:
    """Strip LLM-style prefixes / suffixes to get a concise Bing query."""
    q = raw.strip()
    _PREFIXES = [
        r"^search\s+for\s*:\s*['\"]?",
        r"^search\s*:\s*['\"]?",
        r"^find\s+images?\s+of\s+",
        r"^images?\s+of\s+",
        r"^bing\s+search\s*:\s*",
        r"^query\s*:\s*",
    ]
    for pat in _PREFIXES:
        new_q = re.sub(pat, "", q, flags=re.IGNORECASE).strip()
        if new_q != q:
            q = new_q
            break
    q = q.strip("'\"").strip()
    if len(q) > 80:
        q = " ".join(q.split()[:6])
    return q.strip() or raw.strip()


async def _warmup_bing_session(
    session: aiohttp.ClientSession,
    keyword: str,
    debug: bool = False,
) -> None:
    """Visit a chain of Bing pages to accumulate search cookies."""
    q_encoded = quote(keyword)
    ua = _pick_ua()

    steps = [
        (
            "GET",
            "https://www.bing.com/",
            {
                "User-Agent": ua,
                "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate",
            },
        ),
        (
            "GET",
            f"https://www.bing.com/images/search?q={q_encoded}&form=HDRSC2",
            {
                "User-Agent": ua,
                "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate",
                "Referer": "https://www.bing.com/",
            },
        ),
        (
            "GET",
            (
                f"https://www.bing.com/images/async?q={q_encoded}"
                f"&first=0&count=1&relp=1&scenario=ImageBasicHover"
                f"&ensearch=1&mkt=en-US&safesearch=off"
            ),
            {
                "User-Agent": ua,
                "Referer": (
                    f"https://www.bing.com/images/search?q={q_encoded}&form=HDRSC2"
                ),
                "X-Requested-With": "XMLHttpRequest",
                "Accept-Encoding": "gzip, deflate",
                "Accept-Language": "en-US,en;q=0.9",
            },
        ),
    ]

    for _method, url, hdrs in steps:
        try:
            async with session.get(
                url,
                headers=hdrs,
                timeout=ClientTimeout(total=12),
            ) as resp:
                await resp.text(errors="replace")
            await asyncio.sleep(random.uniform(0.3, 0.8))
        except Exception as e:
            if debug:
                print(f"[warmup] step failed url={url[:60]} err={e}")

    for name, value in _BING_STATIC_COOKIES.items():
        session.cookie_jar.update_cookies(
            {name: value},
            response_url=aiohttp.client_reqrep.URL("https://www.bing.com/"),
        )

    if debug:
        cookie_names = [m.key for m in session.cookie_jar]
        print(f"[warmup] done. cookies={cookie_names}")


def _check_result_relevance(
    urls: List[str],
    keyword: str,
    irrelevant_threshold: float = 0.5,
) -> Tuple[bool, float]:
    """Return ``(is_relevant, irrelevant_ratio)`` for a list of URLs."""
    if not urls:
        return True, 0.0

    irrelevant_count = sum(1 for u in urls if _is_irrelevant_url(u))
    ratio = irrelevant_count / len(urls)
    is_relevant = ratio < irrelevant_threshold
    return is_relevant, ratio


def make_session(
    pool_maxsize: int = 100,
    max_retries: int = 3,
    backoff: float = 0.5,
    headers: Optional[dict] = None,
) -> requests.Session:
    """Create a ``requests.Session`` with retry logic and User-Agent."""
    s = requests.Session()
    retries = Retry(
        total=max_retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD", "OPTIONS"]),
    )
    adapter = HTTPAdapter(
        pool_connections=pool_maxsize,
        pool_maxsize=pool_maxsize,
        max_retries=retries,
    )
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    hdrs = {"User-Agent": _pick_ua(), **_FETCH_HEADERS_BASE}
    if headers:
        hdrs.update(headers)
    s.headers.update(hdrs)
    return s


class _AdaptiveConcurrency:
    """Dynamically adjusts download concurrency based on success rate."""

    def __init__(
        self,
        init_workers: int,
        min_workers: int,
        max_workers: int,
        interval: int = 80, 
    ) -> None:
        self.current = init_workers
        self.min = min_workers
        self.max = max_workers
        self.interval = interval
        self.success = 0
        self.fail = 0
        self._last_adjust_total = 0

    def report(self, ok: bool) -> None:
        if ok:
            self.success += 1
        else:
            self.fail += 1

    def maybe_adjust(self) -> None:
        total = self.success + self.fail
        if total - self._last_adjust_total < self.interval:
            return
        self._last_adjust_total = total
        rate = self.success / total if total else 0
        if rate < 0.50 and self.current > self.min:
            self.current = max(self.min, self.current - 2)
            print(
                f"[async-adapt] success_rate={rate:.2f} "
                f"↓concurrency={self.current}"
            )
        elif rate > 0.80 and self.current < self.max:
            self.current = min(self.max, self.current + 2)
            print(
                f"[async-adapt] success_rate={rate:.2f} "
                f"↑concurrency={self.current}"
            )


class _FetchStats:
    """Collects per-keyword fetch statistics."""

    def __init__(self) -> None:
        self.total_pages = 0
        self.challenge_pages = 0
        self.empty_pages = 0
        self.total_urls_parsed = 0
        self.retry_429 = 0
        self.seen_filtered = 0
        self.irrelevant_pages = 0

    def report_challenge(self) -> None:
        self.total_pages += 1
        self.challenge_pages += 1

    def report_empty(self) -> None:
        self.total_pages += 1
        self.empty_pages += 1

    def report_ok(self, n: int) -> None:
        self.total_pages += 1
        self.total_urls_parsed += n

    def report_429(self) -> None:
        self.retry_429 += 1

    def report_irrelevant(self) -> None:
        self.irrelevant_pages += 1

    def summary(self) -> str:
        avg = self.total_urls_parsed / max(self.total_pages, 1)
        cr = self.challenge_pages / max(self.total_pages, 1)
        return (
            f"pages={self.total_pages} challenge={cr:.2%} "
            f"empty={self.empty_pages} "
            f"irrelevant={self.irrelevant_pages} "
            f"avg_urls={avg:.1f} 429={self.retry_429} "
            f"seen_filtered={self.seen_filtered}"
        )


async def async_fetch_image_urls(
    keyword: str,
    *,
    max_images: int = 300,
    step: int = _DEFAULT_STEP,
    per_page_timeout: float = 12.0,
    max_concurrent_pages: int = 3,
    max_empty_pages: int = 4,
    debug: bool = False,
    session: Optional[aiohttp.ClientSession] = None,
    retry_per_page: int = 3,
    already_seen: Optional[Set[str]] = None,
    max_warmup_retries: int = 2,
) -> List[str]:
    """Sliding-window concurrent Bing image URL scraper."""
    keyword = _clean_query(keyword)
    if debug:
        print(f"[fetch] cleaned keyword='{keyword}'")

    async def _do_fetch(sess: aiohttp.ClientSession) -> List[str]:
        """Internal fetch loop using a warm session."""
        q = quote(keyword)
        found: Set[str] = set()
        fetch_stats = _FetchStats()
        consecutive_empty = 0
        total_empty = 0
        max_total_empty = max_empty_pages * 4
        _next_start = 0
        _challenge_pause = False

        async def _fetch_page(start: int) -> Tuple[List[str], bool]:
            """Fetch one page and return (urls, is_challenge)."""
            url = (
                f"https://www.bing.com/images/async"
                f"?q={q}&first={start}&count={step}&relp={step}"
                f"&scenario=ImageBasicHover&ensearch=1"
                f"&mkt=en-US&safesearch=off"
            )
            hdrs = {
                "User-Agent": _pick_ua(),
                "Referer": (
                    f"https://www.bing.com/images/search?q={q}&form=HDRSC2"
                ),
                "X-Requested-With": "XMLHttpRequest",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate",
            }

            for attempt in range(1, retry_per_page + 2):
                try:
                    async with sess.get(
                        url,
                        headers=hdrs,
                        timeout=ClientTimeout(total=per_page_timeout),
                    ) as resp:
                        if resp.status == 429:
                            fetch_stats.report_429()
                            wait = min(
                                30,
                                2 ** attempt * (1 + random.random() * 0.5),
                            )
                            if debug:
                                print(f"[fetch] 429 start={start} wait={wait:.1f}s")
                            await asyncio.sleep(wait)
                            continue
                        if resp.status != 200:
                            await asyncio.sleep(0.5 * attempt)
                            continue
                        text = await resp.text(errors="replace")

                except asyncio.TimeoutError:
                    await asyncio.sleep(1.0 * attempt)
                    continue
                except (
                    aiohttp.ClientOSError,
                    aiohttp.ServerDisconnectedError,
                    aiohttp.ClientConnectionError,
                ):
                    await asyncio.sleep(0.5 * attempt)
                    continue
                except Exception as e:
                    if debug:
                        print(
                            f"[fetch] exception start={start}: "
                            f"{type(e).__name__}: {e}"
                        )
                    return [], False

                if len(text) < MIN_VALID_HTML_LEN:
                    if debug:
                        snippet = " ".join(text.split())[:200]
                        print(
                            f"[fetch] CHALLENGE start={start} "
                            f"len={len(text)} snippet='{snippet}'"
                        )
                    fetch_stats.report_challenge()
                    await asyncio.sleep(
                        random.uniform(_CHALLENGE_BACKOFF_MIN, _CHALLENGE_BACKOFF_MAX)
                    )
                    return [], True

                urls: List[str] = []
                soup = BeautifulSoup(text, "html.parser")
                for tag in soup.find_all(["a", "div"], class_="iusc"):
                    m_attr = tag.get("m")
                    if not m_attr:
                        continue
                    try:
                        m_json = json.loads(m_attr)
                        img_url = m_json.get("murl") or m_json.get("turl")
                        if img_url and img_url.startswith("http"):
                            urls.append(img_url)
                    except Exception:
                        continue

                if not urls:
                    for m in re.finditer(r'"murl"\s*:\s*"([^"]+)"', text):
                        u = m.group(1)
                        if u.startswith("http"):
                            urls.append(u)

                if urls:
                    is_relevant, irr_ratio = _check_result_relevance(urls, keyword)
                    if not is_relevant:
                        fetch_stats.report_irrelevant()
                        if debug:
                            print(
                                f"[fetch] IRRELEVANT start={start} "
                                f"irr_ratio={irr_ratio:.2%} "
                                f"sample={urls[0][:60]}"
                            )
                        await asyncio.sleep(
                            random.uniform(_PAGE_DELAY_MIN, _PAGE_DELAY_MAX)
                        )
                        return [], False

                if debug and not urls:
                    print(f"[fetch] empty parse start={start} len={len(text)}")

                await asyncio.sleep(random.uniform(_PAGE_DELAY_MIN, _PAGE_DELAY_MAX))

                if urls:
                    fetch_stats.report_ok(len(urls))
                else:
                    fetch_stats.report_empty()
                return urls, False

            return [], False

        pending: Dict[asyncio.Task, int] = {}

        while True:
            if not _challenge_pause:
                while len(pending) < max_concurrent_pages:
                    start = _next_start
                    _next_start += step
                    if start > max_images * 6:
                        break
                    task = asyncio.create_task(_fetch_page(start))
                    pending[task] = start

            if not pending:
                break

            done, _ = await asyncio.wait(
                pending.keys(),
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                start_of_task = pending.pop(task)
                try:
                    page_urls, is_challenge = task.result()
                except Exception:
                    page_urls, is_challenge = [], False

                if is_challenge:
                    _challenge_pause = True
                    if debug:
                        print(f"[fetch] Challenge at start={start_of_task}, pausing.")
                    continue

                _challenge_pause = False

                use_seen_filter = (
                    already_seen is not None and len(already_seen) < MAX_SEEN_CAP
                )

                new_cnt = 0
                for u in page_urls:
                    if use_seen_filter and u in already_seen:
                        fetch_stats.seen_filtered += 1
                        continue
                    if u not in found:
                        found.add(u)
                        if use_seen_filter:
                            already_seen.add(u)
                        new_cnt += 1
                        if len(found) >= max_images:
                            break

                if new_cnt == 0 and not is_challenge:
                    consecutive_empty += 1
                    total_empty += 1
                else:
                    consecutive_empty = 0

                if debug:
                    print(
                        f"[fetch] new={new_cnt} total={len(found)} "
                        f"consec_empty={consecutive_empty} total_empty={total_empty}"
                    )

            if len(found) >= max_images:
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending.keys(), return_exceptions=True)
                break
            if consecutive_empty >= max_empty_pages or total_empty >= max_total_empty:
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending.keys(), return_exceptions=True)
                break

        if debug:
            print(
                f"[fetch] done keyword='{keyword}' "
                f"total={len(found)} stats=({fetch_stats.summary()})"
            )
        return list(found)[:max_images]

    created = False
    if session is None:
        created = True

    for warmup_attempt in range(max_warmup_retries + 1):
        if created or warmup_attempt > 0:
            if warmup_attempt > 0 and debug:
                print(f"[fetch] Resetting session for warmup retry {warmup_attempt}")
            timeout = ClientTimeout(total=per_page_timeout + 5)
            connector = aiohttp.TCPConnector(
                limit=64,
                force_close=(warmup_attempt > 0),
            )
            _session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={"User-Agent": _pick_ua(), **_FETCH_HEADERS_BASE},
                cookie_jar=aiohttp.CookieJar(),
            )
        else:
            _session = session

        try:
            await _warmup_bing_session(_session, keyword, debug=debug)
            result = await _do_fetch(_session)

            if result:
                is_relevant, irr_ratio = _check_result_relevance(result, keyword)
                if not is_relevant and warmup_attempt < max_warmup_retries:
                    if debug:
                        print(
                            f"[fetch] Final results irrelevant "
                            f"(irr_ratio={irr_ratio:.2%}), retrying with fresh session "
                            f"(attempt {warmup_attempt + 1}/{max_warmup_retries})"
                        )
                    if created or warmup_attempt > 0:
                        await _session.close()
                    await asyncio.sleep(random.uniform(2.0, 4.0))
                    continue

            return result

        finally:
            if (created or warmup_attempt > 0) and not _session.closed:
                await _session.close()

    return []


async def async_download_urls(
    urls: List[str],
    save_dir: str,
    *,
    concurrency_initial: int = 12,
    concurrency_min: int = 6,
    concurrency_max: int = 32,
    per_host_limit: int = 4,
    connect_timeout: float = _DOWNLOAD_CONNECT_TIMEOUT,
    read_timeout: float = _DOWNLOAD_READ_TIMEOUT,
    retry: int = 2,
    head_precheck: bool = False,
    debug: bool = False,
    watchdog_interval: int = 30,
    watchdog_stall_sec: int = 90,  
) -> List[str]:
    """Download a deduplicated list of image URLs concurrently."""
    if not urls:
        return []

    os.makedirs(save_dir, exist_ok=True)
    urls = list(dict.fromkeys(urls))

    dynamic_ban_until: Dict[str, float] = {}
    host_fail_counts: Dict[str, int] = defaultdict(int)
    host_cooldowns: Dict[str, float] = {}
    host_sems: Dict[str, asyncio.Semaphore] = {}
    fail_reasons: Counter = Counter()
    saved: List[str] = []
    last_progress_ts = [time.time()]

    host_rapid_fail_until: Dict[str, float] = {}
    host_timeout_counts: Dict[str, int] = defaultdict(int)  
    host_success_counts: Dict[str, int] = defaultdict(int)

    _per_request_timeout = ClientTimeout(
        total=_DOWNLOAD_TOTAL_TIMEOUT,
        connect=connect_timeout,
        sock_connect=connect_timeout,
        sock_read=read_timeout,
    )

    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    connector = aiohttp.TCPConnector(
        limit=concurrency_max * 3,
        limit_per_host=per_host_limit,
        force_close=False,
        ssl=ssl_ctx,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
        keepalive_timeout=30,
    )

    adapt = _AdaptiveConcurrency(concurrency_initial, concurrency_min, concurrency_max)
    queue: asyncio.Queue = asyncio.Queue()
    filtered_count = 0

    _cdn_first: List[str] = []
    _others: List[str] = []
    for u in urls:
        h = urlparse(u).netloc or "unknown"
        if _is_blocked_host(h):
            filtered_count += 1
        elif _is_cdn_host(h):
            _cdn_first.append(u)
        else:
            _others.append(u)
    for u in _cdn_first + _others:
        queue.put_nowait(u)

    total_urls = queue.qsize()
    if debug:
        print(
            f"[dl] total_urls={total_urls} "
            f"(cdn={len(_cdn_first)} other={len(_others)} "
            f"filtered={filtered_count})"
        )

    async with aiohttp.ClientSession(
        timeout=ClientTimeout(total=None),
        connector=connector,
        headers={"User-Agent": _pick_ua(), **_DOWNLOAD_HEADERS_BASE},
    ) as dl_session:

        def _host_sem(host: str) -> asyncio.Semaphore:
            if host not in host_sems:
                lim = _per_host_limit(host, per_host_limit)
                host_sems[host] = asyncio.Semaphore(lim)
            return host_sems[host]

        def _is_banned(host: str) -> bool:
            until = dynamic_ban_until.get(host, 0.0)
            if until and time.time() < until:
                return True
            if host in dynamic_ban_until:
                del dynamic_ban_until[host]
                host_fail_counts[host] = 0
            return False

        def _record_host_failure(host: str, reason: str = "") -> None:
            host_fail_counts[host] += 1
            if host_fail_counts[host] >= _HOST_BAN_THRESHOLD:
                dynamic_ban_until[host] = time.time() + _HOST_BAN_DURATION
                host_fail_counts[host] = 0
                if debug:
                    print(f"[dl] timed-ban host={host} reason={reason}")

        def _is_rapid_failed(host: str) -> bool:
            until = host_rapid_fail_until.get(host, 0.0)
            if until and time.time() < until:
                return True
            if host in host_rapid_fail_until:
                del host_rapid_fail_until[host]
                host_timeout_counts[host] = 0  
            return False

        def _record_host_timeout(host: str) -> None:
            """Count per-host timeouts; trigger rapid-fail when threshold hit."""
            if host_success_counts[host] > 0:
                return
            host_timeout_counts[host] += 1
            if host_timeout_counts[host] >= _HOST_RAPID_FAIL_THRESHOLD:
                host_rapid_fail_until[host] = time.time() + _HOST_RAPID_FAIL_DURATION
                host_timeout_counts[host] = 0
                if debug:
                    print(
                        f"[dl] rapid-fail host={host} "
                        f"for {_HOST_RAPID_FAIL_DURATION:.0f}s"
                    )

        def _record_host_success(host: str) -> None:
            host_success_counts[host] += 1
            host_timeout_counts[host] = 0
            if host in host_rapid_fail_until:
                del host_rapid_fail_until[host]

        async def _download_one(u: str) -> Optional[str]:
            parsed = urlparse(u)
            host = parsed.netloc or "unknown"
            if _is_rapid_failed(host):
                fail_reasons["rapid_fail"] += 1
                return None
            if _is_banned(host):
                fail_reasons["dynamic_ban"] += 1
                return None

            now = time.time()
            cooldown_until = host_cooldowns.get(host, 0.0)
            if now < cooldown_until:
                await asyncio.sleep(min(cooldown_until - now, _HOST_COOLDOWN_MAX))

            sem = _host_sem(host)
            async with sem:
                if _is_rapid_failed(host) or _is_banned(host):
                    fail_reasons["rapid_fail"] += 1
                    return None

                url_hash = hashlib.md5(u.encode()).hexdigest()
                dst = os.path.join(save_dir, f"{url_hash}.jpg")
                if os.path.exists(dst):
                    return dst

                hdrs = {
                    "User-Agent": _pick_ua(),
                    "Referer": "https://www.bing.com/",
                    **_DOWNLOAD_HEADERS_BASE,
                }

                tmp = dst + ".tmp"
                replaced = False
                url_ultimately_failed = False

                for attempt in range(1, retry + 2):
                    try:
                        if head_precheck and attempt == 1:
                            try:
                                async with dl_session.head(
                                    u,
                                    headers=hdrs,
                                    timeout=ClientTimeout(total=5),
                                    ssl=False,
                                ) as h_resp:
                                    if h_resp.status != 200:
                                        fail_reasons[f"head_{h_resp.status}"] += 1
                                        return None
                                    ct = h_resp.headers.get("Content-Type", "")
                                    if "image" not in ct.lower():
                                        fail_reasons["head_not_image"] += 1
                                        return None
                            except Exception:
                                pass

                        async with dl_session.get(
                            u,
                            headers=hdrs,
                            ssl=False,
                            timeout=_per_request_timeout,
                        ) as r:
                            st = r.status
                            if st == 200:
                                ct = r.headers.get("Content-Type", "")
                                if "image" not in ct.lower():
                                    fail_reasons["not_image"] += 1
                                    return None
                                size = 0
                                with open(tmp, "wb") as f:
                                    async for chunk in r.content.iter_chunked(
                                        _DEFAULT_CHUNK
                                    ):
                                        if chunk:
                                            f.write(chunk)
                                            size += len(chunk)
                                if size < _MIN_IMAGE_BYTES:
                                    fail_reasons["too_small"] += 1
                                    try:
                                        os.remove(tmp)
                                    except OSError:
                                        pass
                                    return None
                                os.replace(tmp, dst)
                                replaced = True
                                _record_host_success(host)  # NEW
                                return dst

                            elif st in (301, 302, 303, 307, 308):
                                fail_reasons[f"redirect_{st}"] += 1
                                return None
                            elif st in (403, 401):
                                fail_reasons[f"status_{st}"] += 1
                                _record_host_failure(host, f"status_{st}")
                                return None
                            elif st == 429:
                                wait = min(
                                    _HOST_COOLDOWN_MAX,
                                    2 ** min(attempt, 5) * (1 + random.random() * 0.4),
                                )
                                host_cooldowns[host] = time.time() + wait
                                fail_reasons["status_429"] += 1
                                await asyncio.sleep(min(wait, 8))
                                continue
                            elif st in (503, 502, 500):
                                fail_reasons[f"status_{st}"] += 1
                                await asyncio.sleep(1.5 * attempt)
                                continue
                            else:
                                fail_reasons[f"status_{st}"] += 1
                                return None

                    except asyncio.TimeoutError:
                        fail_reasons["timeout"] += 1
                        _record_host_timeout(host) 
                        if attempt == 1:
                            await asyncio.sleep(random.uniform(0.5, 1.5))
                        else:
                            url_ultimately_failed = True
                            break

                    except (ssl.SSLError, aiohttp.ClientConnectorCertificateError):
                        fail_reasons["SSLError"] += 1
                        return None

                    except (
                        aiohttp.ClientConnectorDNSError,
                        aiohttp.ClientConnectorError,
                    ):
                        fail_reasons["ConnectorError"] += 1
                        if attempt == 1:
                            await asyncio.sleep(0.5)
                        else:
                            url_ultimately_failed = True
                            break

                    except (
                        OSError,
                        aiohttp.ClientOSError,
                        aiohttp.ServerDisconnectedError,
                    ):
                        fail_reasons["OSError"] += 1
                        if attempt <= 2:
                            # Stale keep-alive connection: worth one quick retry.
                            await asyncio.sleep(random.uniform(0.2, 0.8))
                        else:
                            url_ultimately_failed = True
                            break

                    except Exception as e:
                        fail_reasons[type(e).__name__] += 1
                        if attempt <= 2:
                            await asyncio.sleep(0.3 * attempt)
                        else:
                            url_ultimately_failed = True
                            break

                    finally:
                        if not replaced and os.path.exists(tmp):
                            try:
                                os.remove(tmp)
                            except Exception:
                                pass

                if url_ultimately_failed:
                    _record_host_failure(host, "retries_exhausted")
                return None

        async def worker(idx: int) -> None:
            while True:
                if idx >= adapt.current:
                    await asyncio.sleep(0.15)
                    continue
                try:
                    u = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                path = await _download_one(u)
                ok = path is not None and path != ""
                if ok:
                    saved.append(path)
                    last_progress_ts[0] = time.time()
                adapt.report(ok)
                adapt.maybe_adjust()
                queue.task_done()
                if debug and (adapt.success + adapt.fail) % 30 == 0:
                    total_done = adapt.success + adapt.fail
                    rate = adapt.success / total_done if total_done else 0
                    active_rf = sum(
                        1 for v in host_rapid_fail_until.values() if time.time() < v
                    )
                    active_bans = sum(
                        1 for v in dynamic_ban_until.values() if time.time() < v
                    )
                    print(
                        f"[dl] {total_done}/{total_urls} "
                        f"saved={len(saved)} rate={rate:.2f} "
                        f"conc={adapt.current} "
                        f"rapid_fail={active_rf} bans={active_bans} "
                        f"top_fail={fail_reasons.most_common(3)}"
                    )

        async def watchdog() -> None:
            while True:
                await asyncio.sleep(watchdog_interval)
                qsize = queue.qsize()
                if qsize == 0 and (adapt.success + adapt.fail) >= total_urls:
                    return
                elapsed = time.time() - last_progress_ts[0]
                if elapsed > watchdog_stall_sec:
                    active_rf = sum(
                        1 for v in host_rapid_fail_until.values() if time.time() < v
                    )
                    active_bans = sum(
                        1 for v in dynamic_ban_until.values() if time.time() < v
                    )
                    print(
                        f"[dl][WATCHDOG] stall {int(elapsed)}s "
                        f"saved={len(saved)} remaining={qsize} "
                        f"success={adapt.success} fail={adapt.fail} "
                        f"concurrency={adapt.current} "
                        f"rapid_fail={active_rf} active_bans={active_bans} "
                        f"top_fail={fail_reasons.most_common(5)}"
                    )

        workers = [asyncio.create_task(worker(i)) for i in range(concurrency_max)]
        wd_task = asyncio.create_task(watchdog())
        await queue.join()
        for w in workers:
            w.cancel()
        wd_task.cancel()
        await asyncio.gather(*workers, wd_task, return_exceptions=True)

    if debug:
        total = adapt.success + adapt.fail
        rate = adapt.success / total if total else 0
        print(
            f"[dl] done saved={len(saved)} success_rate={rate:.2f} "
            f"fail_reasons={dict(fail_reasons.most_common(8))}"
        )
    return saved


async def async_fetch_and_download(
    query: str,
    save_dir: str,
    *,
    max_images: int = 300,
    step: int = _DEFAULT_STEP,
    max_concurrent_pages: int = 3,
    max_empty_pages: int = 4,
    download_concurrency_initial: int = 12,
    download_concurrency_min: int = 6,
    download_concurrency_max: int = 32,
    per_host_limit: int = 4,
    head_precheck: bool = False,
    debug: bool = False,
    already_seen: Optional[Set[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Fetch image URLs and download them in one call."""
    urls = await async_fetch_image_urls(
        query,
        max_images=max_images,
        step=step,
        max_concurrent_pages=max_concurrent_pages,
        max_empty_pages=max_empty_pages,
        debug=debug,
        already_seen=already_seen,
    )
    if debug:
        print(
            f"[async] fetched={len(urls)} urls "
            f"for query='{_clean_query(query)}'"
        )

    saved = await async_download_urls(
        urls,
        save_dir,
        concurrency_initial=download_concurrency_initial,
        concurrency_min=download_concurrency_min,
        concurrency_max=download_concurrency_max,
        per_host_limit=per_host_limit,
        head_precheck=head_precheck,
        debug=debug,
    )
    if debug:
        print(f"[async] saved={len(saved)} for query='{_clean_query(query)}'")
    return urls, saved


def fetch_image_urls(
    keyword: str,
    max_images: int = 300,
    step: int = _DEFAULT_STEP,
    sleep_min: float = 0.8,
    sleep_max: float = 1.6,
    session=None,
    adaptive: bool = True,
    max_empty_pages: int = 4,
) -> List[str]:
    """Synchronous wrapper around :func:`async_fetch_image_urls`."""
    return asyncio.run(
        async_fetch_image_urls(
            keyword,
            max_images=max_images,
            step=step,
            max_empty_pages=max_empty_pages,
            debug=False,
        )
    )


def fetch_and_download(
    query: str,
    save_dir: str,
    *,
    max_images: int = 120,
    step: int = _DEFAULT_STEP,
    min_kb: int = 0,
    num_workers: int = 12,
    per_host_limit: int = 4,
    sleep_min: float = 0.8,
    sleep_max: float = 1.6,
    session=None,
    processed_urls: Optional[Set[str]] = None,
    use_head_precheck: bool = False,
    adaptive: bool = True,
) -> Tuple[List[str], List[str]]:
    """Synchronous wrapper around :func:`async_fetch_and_download`."""
    return asyncio.run(
        async_fetch_and_download(
            query,
            save_dir,
            max_images=max_images,
            step=step,
            download_concurrency_initial=num_workers,
            download_concurrency_max=num_workers * 2,
            per_host_limit=per_host_limit,
            head_precheck=use_head_precheck,
            debug=True,
        )
    )


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(
        description="Bing no-api image fetch + download v3.5"
    )
    parser.add_argument("--query", required=True)
    parser.add_argument("--out", default="tmp_bing_v3")
    parser.add_argument("--max", type=int, default=300)
    parser.add_argument("--step", type=int, default=_DEFAULT_STEP)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--per_host", type=int, default=4)
    parser.add_argument("--pages", type=int, default=3)
    parser.add_argument("--head_precheck", action="store_true")
    parser.add_argument("--debug", action="store_true")
    a = parser.parse_args()

    pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
    start_ts = time.time()
    urls, saved = asyncio.run(
        async_fetch_and_download(
            a.query,
            a.out,
            max_images=a.max,
            step=a.step,
            max_concurrent_pages=a.pages,
            download_concurrency_initial=a.workers,
            download_concurrency_max=a.workers * 2,
            per_host_limit=a.per_host,
            head_precheck=a.head_precheck,
            debug=a.debug,
        )
    )
    elapsed = time.time() - start_ts
    print(
        f"[done] discovered={len(urls)} saved={len(saved)} "
        f"time={elapsed:.1f}s "
        f"rate={len(saved) / max(elapsed, 1) * 3600:.0f}/hr"
    )