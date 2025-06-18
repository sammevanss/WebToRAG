#!/usr/bin/env python3
"""
run_pipeline.py

Runs the end-to-end RAG (Retrieval-Augmented Generation) pipeline.

Inputs:
- Crawler config: `crawl_and_extract_sections.json` (URLs, rules, domains)
- Sectioned content: `sectioned_pages.json` (produced by crawler)

Outputs:
- Vector file: `rag_vectors.json` (text chunks + embeddings + metadata)

How it works:
1. Loads crawler and model configuration
2. Runs the web crawler script to extract structured page content
3. Loads the crawled output and counts the number of documents
4. Runs the vector generation script to embed content
5. Reports completion time and output file size

Note:
Assumes both `crawl_and_extract_sections.py` and `generate_vectors.py`
are implemented and located in the same directory.

Dependencies:
- Python 3
- sentence-transformers (indirectly via `generate_vectors.py`)
- requests / BeautifulSoup (indirectly via `crawl_and_extract_sections.py`)

Usage:
$ python3 run_pipeline.py
"""

import json
import os
import subprocess
import time

# === Paths & Settings ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = {
    "config_file": os.path.join(BASE_DIR, 'config', 'crawl_and_extract_sections.json'),
    "crawler_script": os.path.join(BASE_DIR, 'crawl_and_extract_sections.py'),
    "vectors_script": os.path.join(BASE_DIR, 'generate_vectors.py'),
    "pages_file": os.path.join(BASE_DIR, 'data', 'sectioned_pages.json'),
    "vectors_output": os.path.join(BASE_DIR, 'data', 'rag_vectors.json'),
    "embedding_model": "all-MiniLM-L6-v2",
    "llm_model": "llama3.2:latest"
}


def format_duration(seconds):
    """Convert seconds to mm:ss format."""
    mins, secs = divmod(int(seconds), 60)
    return f"{mins:02d}:{secs:02d}"


def run_step(label, script_path):
    """Run a Python script and print timing information."""
    print(f"\n{label}")
    print("-" * len(label))
    start = time.time()
    try:
        subprocess.run(['python', script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}")
        print(e)
        exit(1)
    duration = format_duration(time.time() - start)
    print(f"Completed in {duration}")


def load_config(path):
    """Load and return JSON config from file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to read config file: {e}")
        exit(1)


def show_config(crawler_cfg):
    """Display key config values."""
    print("Crawler Configuration")
    print(f"    Base URLs:          {', '.join(crawler_cfg.get('base_urls', []))}")
    print(f"    Allowed file types: {', '.join(crawler_cfg.get('allowed_extensions', []))}")
    print(f"    Whitelist domains:  {', '.join(crawler_cfg.get('whitelist_domains', []))}")
    print(f"    Crawl delay:        {crawler_cfg.get('crawl_delay_seconds', 'N/A')}s")

    print("\nModel Configuration")
    print(f"    Embedding Model:    {CONFIG['embedding_model']}")
    print(f"    Ollama LLM Model:   {CONFIG['llm_model']}")


def count_documents(path):
    """Return number of crawled documents, or 0 if missing/error."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return len(json.load(f))
    except Exception:
        return 0


# === Run Pipeline ===
print("RAG Assistant Pipeline Runner")
print("=" * 29)

crawler_config = load_config(CONFIG['config_file'])
show_config(crawler_config)

print("\nStarting pipeline in 2 seconds...\n")
time.sleep(2)

run_step("Stage 1: Crawling Web Pages", CONFIG['crawler_script'])

num_docs = count_documents(CONFIG['pages_file'])
print(f"\nLoaded {num_docs} sectioned documents")

run_step("Stage 2: Generating Vectors", CONFIG['vectors_script'])

if os.path.exists(CONFIG['vectors_output']):
    size_kb = os.path.getsize(CONFIG['vectors_output']) / 1024
    print(f"\nSaved vectors to: {CONFIG['vectors_output']}")
    print(f"File size:         {size_kb:.2f} KB")
else:
    print("\nError: Vector file not found.")

print("\nPipeline finished successfully.")
