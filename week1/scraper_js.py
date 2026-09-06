"""
JavaScript-capable scraper for Week 1 Day 1.

The original scraper.py uses requests.get(), which only downloads the first HTML
payload. Sites like openai.com render most of their text in the browser with
JavaScript, so that first payload is almost empty.

This module opens a real Chromium browser with Playwright, waits for the page
to finish rendering, then extracts text the same way scraper.py does.

Usage in day1.ipynb — keep everything else the same, only change the import:

    from scraper_js import fetch_website_contents

One-time setup (from the project root, with the .venv active):

    uv run playwright install chromium
"""

from concurrent.futures import ThreadPoolExecutor

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

CHAR_LIMIT = 2_000
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
)


def _clean_text_from_html(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else "No title found"
    if soup.body:
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        text = soup.body.get_text(separator="\n", strip=True)
    else:
        text = ""
    return title, text


def _html_with_playwright(url: str) -> str:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(user_agent=USER_AGENT)
        try:
            # "load" waits for the page load event. Avoid "networkidle" on sites
            # like openai.com that keep analytics/WebSockets open forever.
            page.goto(url, wait_until="load", timeout=60_000)
            page.wait_for_timeout(2_000)
            return page.content()
        finally:
            browser.close()


def _fetch_html(url: str) -> str:
    # Jupyter already runs an asyncio loop. Playwright's sync API cannot run
    # inside that loop, so we move it to a background thread when needed.
    try:
        import asyncio

        asyncio.get_running_loop()
    except RuntimeError:
        return _html_with_playwright(url)

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_html_with_playwright, url).result()


def fetch_website_contents(url: str) -> str:
    """
    Return the title and contents of the website at the given url
    after JavaScript has rendered. Truncate to 2,000 characters.
    """
    html = _fetch_html(url)
    title, text = _clean_text_from_html(html)
    return (title + "\n\n" + text)[:CHAR_LIMIT]


def fetch_website_links(url: str) -> list[str]:
    """Return the links on the website at the given url after JS has rendered."""
    html = _fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    links = [link.get("href") for link in soup.find_all("a")]
    return [link for link in links if link]
