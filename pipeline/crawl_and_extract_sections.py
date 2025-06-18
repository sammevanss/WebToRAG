#!/usr/bin/env python3
"""
crawl_and_extract_sections.py

A scoped web crawler for base URLs, optimized for RAG systems.

This script recursively crawls from configured base URLs, extracts visible page text,
splits it by section headers (h2/h3/h4), removes redundant content, and adds metadata
like last updated dates and resource tags (PDFs, videos, etc.).

Inputs:
- Config file: `crawl_and_extract_sections.json` (base URLs, allowed file types, domain whitelist)

Outputs:
- Crawled content: `sectioned_pages.json` (split by section with metadata)

Usage:
$ python3 crawl_and_extract_sections.py
"""

import os
import re
import json
import time
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# --- Configuration ---
BASE_DIR = os.path.dirname(__file__)
CONFIG_FILE = os.path.join(BASE_DIR, 'config', 'crawl_and_extract_sections.json')
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_FILE = os.path.join(DATA_DIR, 'sectioned_pages.json')

HEADERS = {"User-Agent": "Mozilla/5.0"}

def load_config():
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return (
        config['base_urls'],
        tuple(config['allowed_extensions']),
        set(config.get('whitelist_domains', [])),
        config.get('crawl_delay_seconds', 0.5)
    )

try:
    BASE_URLS, ALLOWED_EXT, WHITELIST_DOMAINS, CRAWL_DELAY = load_config()
except Exception as e:
    print(f"[ERROR] Failed to load config: {e}")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

visited_urls = set()
crawled_data_by_url = {}
crawl_queue = [(url, url) for url in BASE_URLS]

# --- URL Utilities ---

def has_allowed_file_type(url: str) -> bool:
    return url.lower().endswith(ALLOWED_EXT)

def is_within_base_url(base: str, candidate: str) -> bool:
    b, c = urlparse(base), urlparse(candidate)
    return b.netloc == c.netloc and c.path.rstrip('/') and c.path.startswith(b.path.rstrip('/'))

def is_whitelisted_domain(url: str) -> bool:
    domain = urlparse(url).netloc.lower()
    return any(domain.endswith(allowed) for allowed in WHITELIST_DOMAINS)

def should_enqueue(link: str, base_url: str) -> bool:
    return (
        has_allowed_file_type(link)
        and link not in visited_urls
        and not any(link == queued[1] for queued in crawl_queue)
        and (is_within_base_url(base_url, link) or is_whitelisted_domain(link))
    )

# --- Section Processing Helpers ---

def extract_embedded_resources(elem, section):
    for a in elem.find_all("a", href=True):
        href = a['href'].strip().lower()
        full_url = urljoin(section['source_url'], href)
        if href.endswith(".pdf"):
            section["resources"]["pdfs"].add(full_url)
        elif "youtube.com" in href or "vimeo.com" in href:
            section["resources"]["videos"].add(full_url)

def get_last_updated(soup: BeautifulSoup) -> str:
    text = soup.get_text(" ", strip=True)
    match = re.search(r"(Last updated|Page last updated)[^\n:]*[:\s]+([A-Za-z]+\s\d{4}|\d{1,2}\s[A-Za-z]+\s\d{4})", text, re.IGNORECASE)
    return match.group(2) if match else ""

def split_into_sections(soup: BeautifulSoup, url: str) -> list:
    main = soup.find("main") or soup.body
    if not main:
        return []

    sections = []
    current = {
        "heading": "Summary",
        "content": [],
        "resources": {"pdfs": set(), "videos": set()},
        "source_url": url
    }

    for elem in main.descendants:
        if elem.name in ("h2", "h3", "h4"):
            if current["content"]:
                sections.append(current)
            current = {
                "heading": elem.get_text(strip=True),
                "content": [],
                "resources": {"pdfs": set(), "videos": set()},
                "source_url": url
            }
        elif elem.name and elem.name not in ("script", "style", "noscript"):
            extract_embedded_resources(elem, current)
            text = elem.get_text(strip=True)
            if text:
                current["content"].append(text)

    if current["content"]:
        sections.append(current)
    return sections

def deduplicate_and_annotate_sections(sections: list, base_url: str, last_updated: str) -> list:
    seen_texts = set()
    result = []
    for sec in sections:
        text = "\n".join(sec["content"]).strip()
        if text and text not in seen_texts:
            seen_texts.add(text)
            pdfs = sorted(sec["resources"]["pdfs"])
            videos = sorted(sec["resources"]["videos"])
            tags = []
            if pdfs:
                tags.append("pdf")
            if videos:
                tags.append("video")
            result.append({
                "source_url": sec["source_url"],
                "base_url": base_url,
                "heading": sec["heading"],
                "text": text,
                "last_updated": last_updated,
                "tags": tags,
                "resources": {"pdfs": pdfs, "videos": videos}
            })
    return result

def extract_sections_from_page(soup: BeautifulSoup, base_url: str, url: str) -> list:
    sections = split_into_sections(soup, url)
    last_updated = get_last_updated(soup)
    return deduplicate_and_annotate_sections(sections, base_url, last_updated)

def extract_links(soup: BeautifulSoup, base_url: str) -> set:
    return {urljoin(base_url, tag['href'].split('#')[0]) for tag in soup.find_all('a', href=True)}

# --- Output ---

def save_output():
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(crawled_data_by_url, f, ensure_ascii=False, indent=2)
    logging.info("Saved %d pages to %s", len(crawled_data_by_url), OUTPUT_FILE)

# --- Main Crawler ---

def crawl():
    while crawl_queue:
        base_url, url = crawl_queue.pop(0)
        if url in visited_urls:
            continue

        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')

            sections = extract_sections_from_page(soup, base_url, url)
            if sections:
                crawled_data_by_url[url] = sections

            visited_urls.add(url)
            logging.info("Crawled: %s (%d sections)", url, len(sections))

            if is_within_base_url(base_url, url) and not is_whitelisted_domain(url):
                for link in extract_links(soup, url):
                    if should_enqueue(link, base_url):
                        new_base = base_url if is_within_base_url(base_url, link) else link
                        crawl_queue.append((new_base, link))

            time.sleep(CRAWL_DELAY)

        except Exception as e:
            logging.warning("Failed: %s (%s)", url, str(e))

    save_output()

# --- Entry Point ---

def main():
    logging.info("Starting crawl...")
    crawl()

if __name__ == '__main__':
    main()
