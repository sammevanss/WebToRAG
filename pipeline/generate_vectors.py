#!/usr/bin/env python3
"""
generate_vectors.py

Prepares document embeddings for use in a RAG (Retrieval-Augmented Generation) system.

This script processes structured web content into smaller, searchable text chunks and
generates embeddings using a sentence-transformer model. The result is saved to disk
for later use in retrieval-augmented generation.

Inputs:
- Crawled/sectioned content: data/sectioned_pages.json
- Config file: config/generate_vectors.json

Outputs:
- Vectorized output: data/rag_vectors.json

Steps:
1. Load structured pages from disk
2. Split section text into overlapping chunks
3. Embed each chunk using a sentence-transformer
4. Save embedded chunks and metadata to JSON

Dependencies:
- sentence-transformers
- tqdm

Usage:
$ python3 generate_vectors.py
"""

import json
import logging
import os
import re
from typing import List, Dict

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Configuration loading ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'config', 'generate_vectors.json')

try:
    with open(CONFIG_PATH, 'r', encoding='utf-8') as config_file:
        config = json.load(config_file)

    PAGES_PATH = os.path.join(SCRIPT_DIR, 'data', config.get('pages_file', 'sectioned_pages.json'))
    OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'data', config.get('output_file', 'rag_vectors.json'))
    CHUNK_SIZE = config.get('chunk_size', 500)
    CHUNK_OVERLAP = config.get('chunk_overlap', 100)
    EMBED_MODEL_NAME = config.get('embed_model', 'all-MiniLM-L6-v2')
except Exception as e:
    logging.error("Error loading config from %s: %s", CONFIG_PATH, e)
    exit(1)


def sanitize_heading(heading: str) -> str:
    """
    Converts a heading to a lowercase, alphanumeric, underscore-safe string
    suitable for use in identifiers.
    """
    heading = heading.lower().strip()
    heading = re.sub(r'\s+', '_', heading)
    heading = re.sub(r'[^a-z0-9_]', '', heading)
    return heading or "untitled"


def load_pages() -> Dict[str, List[Dict]]:
    """Load structured pages (URL -> list of section dicts) from JSON file."""
    with open(PAGES_PATH, 'r', encoding='utf-8') as pages_file:
        return json.load(pages_file)


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    """Split long text into overlapping chunks."""
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        chunks.append(text[start:start + chunk_size])
    return chunks


def embed_chunks(model, chunks: List[Dict]) -> List[Dict]:
    """Generate embeddings for all text chunks."""
    texts = [chunk['content'] for chunk in chunks]
    vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    for i, vec in enumerate(vectors):
        chunks[i]['embedding'] = vec.tolist()
    return chunks


def run_vector_generator():
    """Main execution logic."""
    logging.info("Loading pages from %s", PAGES_PATH)
    pages = load_pages()
    all_chunks = []

    logging.info("Splitting sections into text chunks...")
    for url, sections in tqdm(pages.items(), desc="Chunking"):
        for section in sections:
            section_text = section.get("text", "").strip()
            if not section_text:
                continue

            heading = section.get("heading", "")
            clean_heading = sanitize_heading(heading)
            source_url = section.get("source_url", url)

            for i, chunk in enumerate(chunk_text(section_text)):
                chunk_id = f"{source_url}#{clean_heading}_{i}"
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "url": source_url,
                    "base_url": section.get("base_url", ""),
                    "heading": heading,
                    "content": chunk,
                    "tags": section.get("tags", []),
                    "resources": section.get("resources", {}),
                    "last_updated": section.get("last_updated", "")
                })

    logging.info("Loading embedding model: %s", EMBED_MODEL_NAME)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    logging.info("Generating embeddings for %d chunks...", len(all_chunks))
    embedded_chunks = embed_chunks(model, all_chunks)

    result = {
        "embedding_model": EMBED_MODEL_NAME,
        "chunks": embedded_chunks
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as output_file:
        json.dump(result, output_file, indent=2)

    logging.info("Saved %d embedded chunks to %s", len(embedded_chunks), OUTPUT_PATH)


if __name__ == '__main__':
    run_vector_generator()
