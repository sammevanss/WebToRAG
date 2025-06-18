# WebAssistantRAG

**WebAssistantRAG** is a modular pipeline for building Retrieval-Augmented Generation (RAG) applications using custom web-sourced content. It includes components for crawling structured content, generating vector embeddings, and interacting with the resulting knowledge base through a command-line interface.

---

## Project Structure

```
project-root/
│
├── pipeline/
│   ├── run_pipeline.py                 # Orchestrates the full pipeline
│   ├── crawl_and_extract_sections.py  # Crawls web pages and extracts structured sections
│   ├── generate_vectors.py            # Converts extracted content to vector embeddings
│   └── config/
│       ├── crawl_and_extract_sections.json
│       └── generate_vectors.json
│
├── user_interfaces/
│   ├── ask_rag_cli.py                 # CLI interface to query the knowledge base
│   └── config/
│       └── llm_config.json
```

---

## Dependencies

Install the required Python packages:

* `requests`
* `beautifulsoup4`
* `tqdm`
* `sentence-transformers`
* `numpy`
* `pandas`
* `ollama`

You can install them all at once:

```
pip install requests beautifulsoup4 tqdm sentence-transformers numpy scikit-learn ollama
```

---

## Usage

### 1. Crawl and Extract Content

Edit `pipeline/config/crawl_and_extract_sections.json` to set your target base URLs and domains, then run:

```
python pipeline/crawl_and_extract_sections.py
```

### 2. Generate Vector Embeddings

Ensure your extraction config is valid, then run:

```
python pipeline/generate_vectors.py
```

### 3. Query the System

Launch the CLI interface to interact with the embedded data:

```
python user_interfaces/ask_rag_cli.py
```

---

## Configuration

* `crawl_and_extract_sections.json`: Controls the scope and depth of web crawling.
* `generate_vectors.json`: Settings for chunking, model choice, and vector storage.
* `llm_config.json`: Configuration for the language model used in RAG.
